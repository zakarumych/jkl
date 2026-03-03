use std::num::NonZero;

use crate::{
    bits::{ReadBits, WriteBits},
    encode::VarCode,
    math::Delta,
    vle,
};

/// A run-length encoded element: a `value` repeated `count` times.
///
/// `Rle` is [`Copy`] when `T` is, and implements [`IntoIterator`] to expand
/// back into individual elements. It also implements [`VarCode`] for
/// bit-level serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rle<T> {
    pub value: T,
    pub count: NonZero<usize>,
}

impl<T> Default for Rle<T>
where
    T: Default,
{
    #[inline]
    fn default() -> Self {
        Rle {
            value: T::default(),
            count: const { NonZero::new(1).unwrap() },
        }
    }
}

impl<T> Delta for Rle<T>
where
    T: Delta,
{
    fn delta(self, base: Self) -> Self {
        Rle {
            value: self.value.delta(base.value),
            count: self.count,
        }
    }

    fn from_delta(base: Self, delta: Self) -> Self {
        Rle {
            value: T::from_delta(base.value, delta.value),
            count: delta.count,
        }
    }
}

impl<T> VarCode for Rle<T>
where
    T: VarCode,
{
    fn var_bit_len(&self) -> usize {
        self.value.var_bit_len() + vle::encode_non_zero_bit_len(self.count.get())
    }

    fn var_write(&self, writer: &mut WriteBits<impl std::io::Write>) -> std::io::Result<()> {
        self.value.var_write(writer)?;
        vle::encode_non_zero(self.count.get(), writer)
    }

    fn var_read(reader: &mut ReadBits<impl std::io::Read>) -> std::io::Result<Self> {
        let value = T::var_read(reader)?;

        let count = vle::decode_non_zero::<usize, _>(reader)?;

        let count = NonZero::new(count).expect("vle::decode_non_zero should never return zero");

        Ok(Rle { value, count })
    }
}

/// Iterator that expands a single [`Rle`] back into `count` copies of its value.
pub struct RleIntoIter<T> {
    value: T,
    count: usize,
}

impl<T> Iterator for RleIntoIter<T>
where
    T: Copy,
{
    type Item = T;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            self.count -= 1;
            Some(self.value)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<T> {
        if n >= self.count {
            self.count = 0;
            None
        } else {
            self.count -= n + 1;
            Some(self.value)
        }
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, T) -> B,
    {
        let mut acc = init;
        for _ in 0..self.count {
            acc = f(acc, self.value);
        }
        acc
    }
}

impl<T> IntoIterator for Rle<T>
where
    T: Copy,
{
    type Item = T;
    type IntoIter = RleIntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        RleIntoIter {
            value: self.value,
            count: self.count.get(),
        }
    }
}

/// Configuration for the RLE encoder.
///
/// Controls the maximum run length and whether runs are split into
/// power-of-two lengths (useful for encodings where power-of-two counts
/// compress more efficiently).
#[derive(Clone, Copy, Debug)]
pub struct RleCfg {
    /// Maximum allowed run length. Runs longer than this are split.
    pub max: usize,
    /// When `true`, each emitted run length is a power of two.
    pub only_power_of_two: bool,
}

impl Default for RleCfg {
    fn default() -> Self {
        RleCfg {
            max: usize::MAX,
            only_power_of_two: false,
        }
    }
}

/// Compresses an iterator of elements into a run-length encoded iterator.
///
/// Consecutive equal elements are collapsed into [`Rle`] runs with no upper
/// bound on run length. Equivalent to `rle_with_cfg(iter, RleCfg::default())`.
pub fn rle<T, I>(iter: I) -> RleIter<T, I::IntoIter>
where
    T: Eq + Copy,
    I: IntoIterator<Item = T>,
{
    rle_with_cfg(iter, RleCfg::default())
}

/// Like [`rle`], but every emitted run length is a power of two.
///
/// Longer runs are split into descending powers of two.
pub fn rle_power_of_two<T, I>(iter: I) -> RleIter<T, I::IntoIter>
where
    T: Eq + Copy,
    I: IntoIterator<Item = T>,
{
    rle_with_cfg(
        iter,
        RleCfg {
            max: usize::MAX,
            only_power_of_two: true,
        },
    )
}

/// Compresses an iterator of elements into a run-length encoded iterator
/// using the provided [`RleCfg`] configuration.
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

/// A lazy iterator adapter that yields [`Rle`] runs from an underlying element iterator.
///
/// Created by [`rle`], [`rle_power_of_two`], or [`rle_with_cfg`].
#[derive(Clone, Debug)]
pub struct RleIter<T, I> {
    last: Option<Rle<T>>,
    next: Option<T>,
    iter: I,
    cfg: RleCfg,
}

fn split_power_of_two(x: NonZero<usize>) -> (NonZero<usize>, Option<NonZero<usize>>) {
    if x.is_power_of_two() {
        (x, None)
    } else {
        let last_power_of_two = ((x.get() / 2) + 1).next_power_of_two();
        let rest = x.get() - last_power_of_two;

        (
            NonZero::new(last_power_of_two).expect("last_power_of_two should never be zero"),
            Some(NonZero::new(rest).expect("rest should never be zero if x is not a power of two")),
        )
    }
}

fn flush_rle<T: Copy>(rle: Rle<T>, only_power_of_two: bool) -> (Rle<T>, Option<Rle<T>>) {
    if !only_power_of_two || rle.count.is_power_of_two() {
        (rle, None)
    } else {
        // Or in power-of-two's
        let (last_power_of_two, rest) = split_power_of_two(rle.count);

        (
            Rle {
                value: rle.value,
                count: last_power_of_two,
            },
            rest.map(|count| Rle {
                value: rle.value,
                count,
            }),
        )
    }
}

fn fold_rle<T: Copy, B>(
    rle: Rle<T>,
    only_power_of_two: bool,
    init: B,
    mut f: impl FnMut(B, Rle<T>) -> B,
) -> B {
    if !only_power_of_two {
        return f(init, rle);
    }

    let mut acc = init;

    let value = rle.value;
    let mut count = rle.count;

    loop {
        // Or in power-of-two's
        let (last_power_of_two, rest) = split_power_of_two(count);

        acc = f(
            acc,
            Rle {
                value: value,
                count: last_power_of_two,
            },
        );

        match rest {
            None => break,
            Some(rest) => count = rest,
        }
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
                            count: const { NonZero::new(1).unwrap() },
                        });
                        self.next = self.iter.next();
                    }
                    Some(mut rle) if rle.value == next && rle.count.get() < self.cfg.max => {
                        rle.count = NonZero::new(rle.count.get() + 1).expect(
                            "count is less than another usize, so it should never overflow",
                        );
                        self.last = Some(rle);
                        self.next = self.iter.next();
                    }
                    Some(rle) => {
                        let (ret, keep) = flush_rle(rle, self.cfg.only_power_of_two);
                        match keep {
                            None => {
                                self.last = Some(Rle {
                                    value: next,
                                    count: const { NonZero::new(1).unwrap() },
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
                        count: const { NonZero::new(1).unwrap() },
                    });
                }
                Some(mut rle) if rle.value == next && rle.count.get() < cfg.max => {
                    rle.count = NonZero::new(rle.count.get() + 1)
                        .expect("count is less than another usize, so it should never overflow");
                    last = Some(rle);
                }
                Some(rle) => {
                    acc = fold_rle(rle, cfg.only_power_of_two, acc, &mut f);

                    last = Some(Rle {
                        value: next,
                        count: const { NonZero::new(1).unwrap() },
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
