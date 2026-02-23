use crate::{
    bits::{ReadBits, WriteBits},
    encode::Encode,
    vle,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rle<T> {
    pub value: T,
    pub count: u32,
}

impl<T> Encode for Rle<T>
where
    T: Encode,
{
    fn write(&self, writer: &mut WriteBits<impl std::io::Write>) -> std::io::Result<()> {
        self.value.write(writer)?;
        vle::encode(self.count, writer)
    }

    fn read(reader: &mut ReadBits<impl std::io::Read>) -> std::io::Result<Self> {
        let value = T::read(reader)?;

        let count = vle::decode::<u32, _>(reader)?;

        Ok(Rle { value, count })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RleCfg {
    pub max: u32,
    pub only_power_of_two: bool,
}

impl Default for RleCfg {
    fn default() -> Self {
        RleCfg {
            max: u32::MAX,
            only_power_of_two: false,
        }
    }
}

pub fn rle<T, I>(iter: I) -> RleIter<T, I::IntoIter>
where
    T: Eq + Copy,
    I: IntoIterator<Item = T>,
{
    rle_with_cfg(iter, RleCfg::default())
}

pub fn rle_power_of_two<T, I>(iter: I) -> RleIter<T, I::IntoIter>
where
    T: Eq + Copy,
    I: IntoIterator<Item = T>,
{
    rle_with_cfg(
        iter,
        RleCfg {
            max: u32::MAX,
            only_power_of_two: true,
        },
    )
}

pub fn rle_with_cfg<T, I>(iter: I, cfg: RleCfg) -> RleIter<T, I::IntoIter>
where
    T: Eq + Copy,
    I: IntoIterator<Item = T>,
{
    let mut iter = iter.into_iter();

    RleIter {
        last: None,
        next: iter.next(),
        iter,
        cfg,
    }
}

#[derive(Clone, Debug)]
pub struct RleIter<T, I> {
    last: Option<Rle<T>>,
    next: Option<T>,
    iter: I,
    cfg: RleCfg,
}

fn prev_power_of_two(x: u32) -> u32 {
    (x / 2 + 1).next_power_of_two()
}

fn flush_rle<T: Copy>(rle: Rle<T>, only_power_of_two: bool) -> (Rle<T>, Option<Rle<T>>) {
    debug_assert_ne!(rle.count, 0);

    if !only_power_of_two || rle.count.is_power_of_two() {
        (rle, None)
    } else {
        // Or in power-of-two's
        let prev_power_of_two = prev_power_of_two(rle.count);

        (
            Rle {
                value: rle.value,
                count: prev_power_of_two,
            },
            Some(Rle {
                value: rle.value,
                count: rle.count - prev_power_of_two,
            }),
        )
    }
}

fn fold_rle<T: Copy, B>(
    mut rle: Rle<T>,
    only_power_of_two: bool,
    init: B,
    mut f: impl FnMut(B, Rle<T>) -> B,
) -> B {
    debug_assert_ne!(rle.count, 0);

    if !only_power_of_two {
        return f(init, rle);
    }

    let mut acc = init;

    while rle.count > 0 {
        // Or in power-of-two's
        let prev_power_of_two = prev_power_of_two(rle.count);

        acc = f(
            acc,
            Rle {
                value: rle.value,
                count: prev_power_of_two,
            },
        );

        rle.count -= prev_power_of_two;
    }

    acc
}

impl<T, I> RleIter<T, I>
where
    I: Iterator<Item = T>,
    T: Eq + Copy,
{
}

impl<T, I> Iterator for RleIter<T, I>
where
    I: Iterator<Item = T>,
    T: Eq + Copy,
{
    type Item = Rle<T>;

    fn next(&mut self) -> Option<Rle<T>> {
        loop {
            match self.next.take() {
                None => {
                    // Iterator was exhausted.
                    // Flush last accumulated RLE
                    match self.last.take() {
                        None => return None, // No more,
                        Some(rle) => {
                            let (ret, keep) = flush_rle(rle, self.cfg.only_power_of_two);
                            self.last = keep;

                            // Return biggest power of two
                            return Some(ret);
                        }
                    }
                }
                Some(next) => match self.last.take() {
                    None => {
                        self.last = Some(Rle {
                            value: next,
                            count: 1,
                        });
                        self.next = self.iter.next();
                    }
                    Some(mut rle) if rle.value == next && rle.count < self.cfg.max => {
                        rle.count += 1;
                        self.last = Some(rle);
                        self.next = self.iter.next();
                    }
                    Some(rle) => {
                        let (ret, keep) = flush_rle(rle, self.cfg.only_power_of_two);
                        match keep {
                            None => {
                                self.last = Some(Rle {
                                    value: next,
                                    count: 1,
                                });
                                self.next = self.iter.next();
                            }
                            Some(keep) => {
                                self.next = Some(next);
                                self.last = Some(keep);
                            }
                        }
                        return Some(ret);
                    }
                },
            }
        }
    }

    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        // `fn fold()` can be implemented less repeated checks than looping over `fn next()`

        let mut acc = init;
        let RleIter {
            mut last,
            next,
            iter,
            cfg,
        } = self;

        let mut process = |mut acc: B, next| -> B {
            match last.take() {
                None => {
                    last = Some(Rle {
                        value: next,
                        count: 1,
                    });
                }
                Some(mut rle) if rle.value == next && rle.count < cfg.max => {
                    rle.count += 1;
                    last = Some(rle);
                }
                Some(rle) => {
                    acc = fold_rle(rle, cfg.only_power_of_two, acc, &mut f);

                    last = Some(Rle {
                        value: next,
                        count: 1,
                    });
                }
            }
            acc
        };

        if let Some(next) = next {
            acc = process(acc, next);
        }

        acc = iter.fold(acc, |acc, next| process(acc, next));

        if let Some(rle) = last {
            acc = fold_rle(rle, cfg.only_power_of_two, acc, f);
        }

        acc
    }
}
