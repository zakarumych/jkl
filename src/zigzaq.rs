pub trait ZigZag: Copy + 'static {
    type Output: Copy + 'static;

    fn zigzag(self) -> Self::Output;
    fn zagzig(u: Self::Output) -> Self;
}

macro_rules! impl_zig_zag_noop {
    ($($t:ty),* $(,)?) => {
        $(
            impl ZigZag for $t {
                type Output = $t;

                #[inline]
                fn zigzag(self) -> $t {
                    self
                }

                #[inline]
                fn zagzig(u: $t) -> $t {
                    u
                }
            }
        )*
    };
}

macro_rules! impl_zig_zag_signed {
    ($($s:ty as $u:ty),* $(,)?) => {
        $(
            impl ZigZag for $s {
                type Output = $u;

                #[inline]
                fn zigzag(self) -> $u {
                    let shift = <$s>::BITS.saturating_sub(1);
                    let v = (self << 1) ^ (self >> shift);
                    v as $u
                }

                #[inline]
                fn zagzig(u: $u) -> $s {
                    let a = (u >> 1) as $s;
                    let b = -((u & 1) as $s);
                    a ^ b
                }
            }
        )*
    };
}

impl_zig_zag_noop!(u8, u16, u32, u64, u128, usize);
impl_zig_zag_signed!(
    i8 as u8,
    i16 as u16,
    i32 as u32,
    i64 as u64,
    i128 as u128,
    isize as usize,
);
