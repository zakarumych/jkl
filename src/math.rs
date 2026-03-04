//! Linear algebra primitives, color types, and bit-manipulation utilities.
//!
//! Provides [`Vec2`], [`Vec3`], and [`Vec4`] floating-point vectors with
//! swizzle accessors, scalar-valued color types ([`R8U`], [`R32F`], [`Rg8U`],
//! [`Rg32F`], [`Rgb8U`], [`Rgb32F`], [`Rgba8U`], [`Rgba32F`], [`Rgb565`],
//! [`Yiq32F`]), axis-aligned bounding regions ([`Region3`], [`Rect`]),
//! bit-interleaving helpers, and PCA-based axis estimation.

use std::{
    convert::Infallible,
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, Sub, SubAssign},
};

/// Linearly interpolates between `a` and `b` by factor `t`.
///
/// Returns exactly `a` when `t == 0.0` and exactly `b` when `t == 1.0`.
/// The implementation is monotonic and avoids catastrophic cancellation.
#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    // This lerp is monotonic and produces exactly a for t = 0 and b for t = 1.
    // If t is constant the branch will be optimized out.

    if t <= 0.5 {
        (b - a).mul_add(t, a)
    } else {
        (a - b).mul_add(1.0 - t, b)
    }
}

/// Trait to create additive identity element.
pub trait Zero {
    fn zero() -> Self;

    fn is_zero(&self) -> bool;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }

    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

impl Zero for u32 {
    fn zero() -> Self {
        0
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl Zero for i32 {
    fn zero() -> Self {
        0
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl Zero for usize {
    fn zero() -> Self {
        0
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }
}

/// Trait to create multiplicative identity element.
pub trait One {
    fn one() -> Self;

    fn is_one(&self) -> bool;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }

    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl One for u32 {
    fn one() -> Self {
        1
    }

    fn is_one(&self) -> bool {
        *self == 1
    }
}

impl One for i32 {
    fn one() -> Self {
        1
    }

    fn is_one(&self) -> bool {
        *self == 1
    }
}

impl One for usize {
    fn one() -> Self {
        1
    }

    fn is_one(&self) -> bool {
        *self == 1
    }
}

/// Delta allows calculating "difference" between base and current value,
/// and reconstructing current value from base and delta.
///
/// Computed deltas should reduce the entropy of the data, so that they can be efficiently compressed.
pub trait Delta: Ord + Sized {
    /// Calculate the delta between self and base.
    /// Base must be less than or equal to self.
    ///
    /// Using base that is greater than self may produce inadequate deltas
    /// or even panic in debug mode.
    fn delta(self, base: Self) -> Self;

    /// Reconstruct the original value from base and delta.
    fn from_delta(base: Self, delta: Self) -> Self;
}

macro_rules! impl_delta_for_numeric {
    ($($num:ty)*) => {
        $(
            impl Delta for $num {
                #[inline(always)]
                fn delta(self, base: Self) -> Self {
                    self - base
                }

                #[inline(always)]
                fn from_delta(base: Self, delta: Self) -> Self {
                    base + delta
                }
            }
        )*
    };
}

impl_delta_for_numeric!(u8 u16 u32 u64 i8 i16 i32 i64);

/// A 2D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec2([f32; 2]);

impl Zero for Vec2 {
    fn zero() -> Self {
        Vec2::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == [0.0; 2]
    }
}

impl Add for Vec2 {
    type Output = Vec2;

    #[inline(always)]
    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2([self.x() + rhs.x(), self.y() + rhs.y()])
    }
}

impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.x();
        self.0[1] += rhs.y();
    }
}

impl Sub for Vec2 {
    type Output = Vec2;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2([self.x() - rhs.x(), self.y() - rhs.y()])
    }
}

impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.x();
        self.0[1] -= rhs.y();
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Vec2 {
        Vec2([self.x() * rhs, self.y() * rhs])
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
    }
}

impl Div<f32> for Vec2 {
    type Output = Vec2;

    #[inline(always)]
    fn div(self, rhs: f32) -> Vec2 {
        Vec2([self.x() / rhs, self.y() / rhs])
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.0[0] /= rhs;
        self.0[1] /= rhs;
    }
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Vec2 = Vec2([0.0, 0.0]);

    /// Creates a new `Vec2` from individual components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Vec2([x, y])
    }

    /// Creates a `Vec2` with all components set to `value`.
    #[inline(always)]
    pub const fn splat(value: f32) -> Self {
        Vec2([value, value])
    }

    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    pub const fn dot(self, rhs: Vec2) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y()
    }

    /// Returns the Euclidean length of this vector.
    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the squared length of this vector.
    #[inline(always)]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns the unit-length direction of this vector, or `(1, 0)` if zero.
    #[inline(always)]
    pub fn norm(self) -> Self {
        let length = self.length();
        if length != 0.0 {
            self / length
        } else {
            Vec2([1.0, 0.0])
        }
    }

    /// Returns the x component.
    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    /// Returns the y component.
    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub const fn xx(&self) -> Vec2 {
        Vec2([self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xy(&self) -> Vec2 {
        Vec2([self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yx(&self) -> Vec2 {
        Vec2([self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yy(&self) -> Vec2 {
        Vec2([self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xxx(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxy(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xyx(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyy(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yxx(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxy(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yyx(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyy(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xxxx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxxy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xxyx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xxyy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xyxx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xyxy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xyyx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyyy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yxxx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxxy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yxyx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yxyy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yyxx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yyxy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yyyx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyyy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.y()])
    }

    /// Extends this `Vec2` with a z component, producing a `Vec3`.
    #[inline(always)]
    pub const fn with_z(&self, z: f32) -> Vec3 {
        Vec3([self.x(), self.y(), z])
    }

    /// Extends this `Vec2` with z and w components, producing a `Vec4`.
    #[inline(always)]
    pub const fn with_zw(&self, z: f32, w: f32) -> Vec4 {
        Vec4([self.x(), self.y(), z, w])
    }

    /// Linearly interpolates between two `Vec2` values component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec2([lerp(a.x(), b.x(), t), lerp(a.y(), b.y(), t)])
    }
}

/// A 3D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec3([f32; 3]);

impl Zero for Vec3 {
    fn zero() -> Self {
        Vec3::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == [0.0; 3]
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3([self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z()])
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.x();
        self.0[1] += rhs.y();
        self.0[2] += rhs.z();
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3([self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z()])
    }
}

impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.x();
        self.0[1] -= rhs.y();
        self.0[2] -= rhs.z();
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Vec3 {
        Vec3([self.x() * rhs, self.y() * rhs, self.z() * rhs])
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn div(self, rhs: f32) -> Vec3 {
        Vec3([self.x() / rhs, self.y() / rhs, self.z() / rhs])
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
    }
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3([0.0, 0.0, 0.0]);

    /// Creates a new `Vec3` from individual components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }

    /// Creates a `Vec3` with all components set to `value`.
    pub const fn splat(value: f32) -> Self {
        Vec3([value, value, value])
    }

    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    pub const fn dot(self, rhs: Vec3) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
    }

    /// Returns the Euclidean length of this vector.
    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the squared length of this vector.
    #[inline(always)]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns the unit-length direction of this vector, or `(1, 0, 0)` if zero.
    #[inline(always)]
    pub fn norm(self) -> Self {
        let length = self.length();
        if length != 0.0 {
            self / length
        } else {
            Vec3([1.0, 0.0, 0.0])
        }
    }

    /// Returns the x component.
    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    /// Returns the y component.
    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    /// Returns the z component.
    #[inline(always)]
    pub const fn z(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub const fn xx(&self) -> Vec2 {
        Vec2([self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xy(&self) -> Vec2 {
        Vec2([self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xz(&self) -> Vec2 {
        Vec2([self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yx(&self) -> Vec2 {
        Vec2([self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yy(&self) -> Vec2 {
        Vec2([self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yz(&self) -> Vec2 {
        Vec2([self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zx(&self) -> Vec2 {
        Vec2([self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zy(&self) -> Vec2 {
        Vec2([self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zz(&self) -> Vec2 {
        Vec2([self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xxx(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxy(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xxz(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xyx(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyy(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xyz(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xzx(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xzy(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xzz(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yxx(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxy(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yxz(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yyx(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyy(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yyz(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yzx(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yzy(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yzz(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zxx(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zxy(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zxz(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zyx(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zyy(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zyz(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zzx(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zzy(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zzz(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xxxx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxxy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xxxz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xxyx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xxyy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xxyz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xxzx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xxzy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xxzz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xyxx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xyxy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xyxz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xyyx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyyy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xyyz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xyzx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xyzy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xyzz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xzxx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xzxy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xzxz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xzyx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xzyy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xzyz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xzzx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xzzy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xzzz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yxxx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxxy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yxxz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yxyx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yxyy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yxyz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yxzx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yxzy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yxzz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yyxx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yyxy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yyxz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yyyx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyyy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yyyz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yyzx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yyzy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yyzz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yzxx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yzxy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yzxz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yzyx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yzyy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yzyz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yzzx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yzzy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yzzz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zxxx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zxxy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zxxz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zxyx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zxyy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zxyz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zxzx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zxzy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zxzz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zyxx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zyxy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zyxz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zyyx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zyyy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zyyz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zyzx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zyzy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zyzz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zzxx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zzxy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zzxz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zzyx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zzyy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zzyz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zzzx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zzzy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zzzz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.z()])
    }

    /// Extends this `Vec3` with a w component, producing a `Vec4`.
    #[inline(always)]
    pub const fn with_w(&self, w: f32) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), w])
    }

    /// Linearly interpolates between two `Vec3` values component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec3([
            lerp(a.x(), b.x(), t),
            lerp(a.y(), b.y(), t),
            lerp(a.z(), b.z(), t),
        ])
    }
}

/// A 4D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec4([f32; 4]);

impl Zero for Vec4 {
    fn zero() -> Self {
        Vec4::ZERO
    }

    fn is_zero(&self) -> bool {
        self.0 == [0.0; 4]
    }
}

impl Add for Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn add(self, rhs: Vec4) -> Vec4 {
        Vec4([
            self.x() + rhs.x(),
            self.y() + rhs.y(),
            self.z() + rhs.z(),
            self.w() + rhs.w(),
        ])
    }
}

impl AddAssign for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.x();
        self.0[1] += rhs.y();
        self.0[2] += rhs.z();
        self.0[3] += rhs.w();
    }
}

impl Sub for Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec4([
            self.x() - rhs.x(),
            self.y() - rhs.y(),
            self.z() - rhs.z(),
            self.w() - rhs.w(),
        ])
    }
}

impl SubAssign for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.x();
        self.0[1] -= rhs.y();
        self.0[2] -= rhs.z();
        self.0[3] -= rhs.w();
    }
}

impl Mul<f32> for Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Vec4 {
        Vec4([
            self.x() * rhs,
            self.y() * rhs,
            self.z() * rhs,
            self.w() * rhs,
        ])
    }
}

impl MulAssign<f32> for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
    }
}

impl Div<f32> for Vec4 {
    type Output = Vec4;

    #[inline(always)]
    fn div(self, rhs: f32) -> Vec4 {
        Vec4([
            self.x() / rhs,
            self.y() / rhs,
            self.z() / rhs,
            self.w() / rhs,
        ])
    }
}

impl DivAssign<f32> for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
        self.0[3] /= rhs;
    }
}

impl Vec4 {
    /// The zero vector.
    pub const ZERO: Vec4 = Vec4([0.0, 0.0, 0.0, 0.0]);

    /// Creates a new `Vec4` from individual components.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4([x, y, z, w])
    }

    /// Creates a `Vec4` with all components set to `value`.
    #[inline(always)]
    pub const fn splat(value: f32) -> Self {
        Vec4([value, value, value, value])
    }

    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    pub const fn dot(self, rhs: Vec4) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z() + self.w() * rhs.w()
    }

    /// Returns the Euclidean length of this vector.
    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the squared length of this vector.
    #[inline(always)]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// Returns the unit-length direction of this vector, or `(1, 0, 0, 0)` if zero.
    #[inline(always)]
    pub fn norm(self) -> Self {
        let length = self.length();
        if length != 0.0 {
            self / length
        } else {
            Vec4([1.0, 0.0, 0.0, 0.0])
        }
    }

    /// Returns the x component.
    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    /// Returns the y component.
    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    /// Returns the z component.
    #[inline(always)]
    pub const fn z(&self) -> f32 {
        self.0[2]
    }

    /// Returns the w component.
    #[inline(always)]
    pub const fn w(&self) -> f32 {
        self.0[3]
    }

    #[inline(always)]
    pub const fn xx(&self) -> Vec2 {
        Vec2([self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xy(&self) -> Vec2 {
        Vec2([self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xz(&self) -> Vec2 {
        Vec2([self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xw(&self) -> Vec2 {
        Vec2([self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn yx(&self) -> Vec2 {
        Vec2([self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yy(&self) -> Vec2 {
        Vec2([self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yz(&self) -> Vec2 {
        Vec2([self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yw(&self) -> Vec2 {
        Vec2([self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zx(&self) -> Vec2 {
        Vec2([self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zy(&self) -> Vec2 {
        Vec2([self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zz(&self) -> Vec2 {
        Vec2([self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zw(&self) -> Vec2 {
        Vec2([self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wy(&self) -> Vec2 {
        Vec2([self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wz(&self) -> Vec2 {
        Vec2([self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn ww(&self) -> Vec2 {
        Vec2([self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn xxx(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxy(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xxz(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xxw(&self) -> Vec3 {
        Vec3([self.x(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn xyx(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyy(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xyz(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xyw(&self) -> Vec3 {
        Vec3([self.x(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn xzx(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xzy(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xzz(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xzw(&self) -> Vec3 {
        Vec3([self.x(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn xwx(&self) -> Vec3 {
        Vec3([self.x(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn xwy(&self) -> Vec3 {
        Vec3([self.x(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn xwz(&self) -> Vec3 {
        Vec3([self.x(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn xww(&self) -> Vec3 {
        Vec3([self.x(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn yxx(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxy(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yxz(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yxw(&self) -> Vec3 {
        Vec3([self.y(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn yyx(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyy(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yyz(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yyw(&self) -> Vec3 {
        Vec3([self.y(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn yzx(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yzy(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yzz(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yzw(&self) -> Vec3 {
        Vec3([self.y(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn ywx(&self) -> Vec3 {
        Vec3([self.y(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn ywy(&self) -> Vec3 {
        Vec3([self.y(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn ywz(&self) -> Vec3 {
        Vec3([self.y(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn yww(&self) -> Vec3 {
        Vec3([self.y(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn zxx(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zxy(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zxz(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zxw(&self) -> Vec3 {
        Vec3([self.z(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn zyx(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zyy(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zyz(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zyw(&self) -> Vec3 {
        Vec3([self.z(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zzx(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zzy(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zzz(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zzw(&self) -> Vec3 {
        Vec3([self.z(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn zwx(&self) -> Vec3 {
        Vec3([self.z(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn zwy(&self) -> Vec3 {
        Vec3([self.z(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn zwz(&self) -> Vec3 {
        Vec3([self.z(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn zww(&self) -> Vec3 {
        Vec3([self.z(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn wxx(&self) -> Vec3 {
        Vec3([self.w(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn wxy(&self) -> Vec3 {
        Vec3([self.w(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn wxz(&self) -> Vec3 {
        Vec3([self.w(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn wxw(&self) -> Vec3 {
        Vec3([self.w(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn wyx(&self) -> Vec3 {
        Vec3([self.w(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn wyy(&self) -> Vec3 {
        Vec3([self.w(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn wyz(&self) -> Vec3 {
        Vec3([self.w(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn wyw(&self) -> Vec3 {
        Vec3([self.w(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn wzx(&self) -> Vec3 {
        Vec3([self.w(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn wzy(&self) -> Vec3 {
        Vec3([self.w(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn wzz(&self) -> Vec3 {
        Vec3([self.w(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn wzw(&self) -> Vec3 {
        Vec3([self.w(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wwx(&self) -> Vec3 {
        Vec3([self.w(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn wwy(&self) -> Vec3 {
        Vec3([self.w(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wwz(&self) -> Vec3 {
        Vec3([self.w(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn www(&self) -> Vec3 {
        Vec3([self.w(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn xxxx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xxxy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xxxz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xxxw(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn xxyx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xxyy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xxyz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xxyw(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn xxzx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xxzy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xxzz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xxzw(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn xxwx(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn xxwy(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn xxwz(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn xxww(&self) -> Vec4 {
        Vec4([self.x(), self.x(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn xyxx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xyxy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xyxz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xyxw(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn xyyx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xyyy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xyyz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xyyw(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn xyzx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xyzy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xyzz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xyzw(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn xywx(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn xywy(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn xywz(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn xyww(&self) -> Vec4 {
        Vec4([self.x(), self.y(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn xzxx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xzxy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xzxz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xzxw(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn xzyx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xzyy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xzyz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xzyw(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn xzzx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xzzy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xzzz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xzzw(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn xzwx(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn xzwy(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn xzwz(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn xzww(&self) -> Vec4 {
        Vec4([self.x(), self.z(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn xwxx(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn xwxy(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn xwxz(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn xwxw(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn xwyx(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn xwyy(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn xwyz(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn xwyw(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn xwzx(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn xwzy(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn xwzz(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn xwzw(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn xwwx(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn xwwy(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn xwwz(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn xwww(&self) -> Vec4 {
        Vec4([self.x(), self.w(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn yxxx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yxxy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yxxz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yxxw(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn yxyx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yxyy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yxyz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yxyw(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn yxzx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yxzy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yxzz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yxzw(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn yxwx(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn yxwy(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn yxwz(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn yxww(&self) -> Vec4 {
        Vec4([self.y(), self.x(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn yyxx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yyxy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yyxz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yyxw(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn yyyx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yyyy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yyyz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yyyw(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn yyzx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yyzy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yyzz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yyzw(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn yywx(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn yywy(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn yywz(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn yyww(&self) -> Vec4 {
        Vec4([self.y(), self.y(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn yzxx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn yzxy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn yzxz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn yzxw(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn yzyx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn yzyy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn yzyz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn yzyw(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn yzzx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn yzzy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn yzzz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn yzzw(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn yzwx(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn yzwy(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn yzwz(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn yzww(&self) -> Vec4 {
        Vec4([self.y(), self.z(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn ywxx(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn ywxy(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn ywxz(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn ywxw(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn ywyx(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn ywyy(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn ywyz(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn ywyw(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn ywzx(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn ywzy(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn ywzz(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn ywzw(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn ywwx(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn ywwy(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn ywwz(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn ywww(&self) -> Vec4 {
        Vec4([self.y(), self.w(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn zxxx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zxxy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zxxz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zxxw(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn zxyx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zxyy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zxyz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zxyw(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zxzx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zxzy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zxzz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zxzw(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn zxwx(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn zxwy(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn zxwz(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn zxww(&self) -> Vec4 {
        Vec4([self.z(), self.x(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn zyxx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zyxy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zyxz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zyxw(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn zyyx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zyyy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zyyz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zyyw(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zyzx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zyzy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zyzz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zyzw(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn zywx(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn zywy(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn zywz(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn zyww(&self) -> Vec4 {
        Vec4([self.z(), self.y(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn zzxx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zzxy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zzxz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zzxw(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn zzyx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zzyy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zzyz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zzyw(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zzzx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zzzy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zzzz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zzzw(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn zzwx(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn zzwy(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn zzwz(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn zzww(&self) -> Vec4 {
        Vec4([self.z(), self.z(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn zwxx(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn zwxy(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn zwxz(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn zwxw(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn zwyx(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn zwyy(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn zwyz(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn zwyw(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn zwzx(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn zwzy(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn zwzz(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn zwzw(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn zwwx(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn zwwy(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn zwwz(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn zwww(&self) -> Vec4 {
        Vec4([self.z(), self.w(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn wxxx(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn wxxy(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn wxxz(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn wxxw(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn wxyx(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn wxyy(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn wxyz(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn wxyw(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn wxzx(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn wxzy(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn wxzz(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn wxzw(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wxwx(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn wxwy(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wxwz(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn wxww(&self) -> Vec4 {
        Vec4([self.w(), self.x(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn wyxx(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn wyxy(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn wyxz(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn wyxw(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn wyyx(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn wyyy(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn wyyz(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn wyyw(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn wyzx(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn wyzy(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn wyzz(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn wyzw(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wywx(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn wywy(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wywz(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn wyww(&self) -> Vec4 {
        Vec4([self.w(), self.y(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn wzxx(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn wzxy(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn wzxz(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn wzxw(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn wzyx(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn wzyy(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn wzyz(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn wzyw(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn wzzx(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn wzzy(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn wzzz(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn wzzw(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wzwx(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn wzwy(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wzwz(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn wzww(&self) -> Vec4 {
        Vec4([self.w(), self.z(), self.w(), self.w()])
    }

    #[inline(always)]
    pub const fn wwxx(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.x(), self.x()])
    }

    #[inline(always)]
    pub const fn wwxy(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.x(), self.y()])
    }

    #[inline(always)]
    pub const fn wwxz(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.x(), self.z()])
    }

    #[inline(always)]
    pub const fn wwxw(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.x(), self.w()])
    }

    #[inline(always)]
    pub const fn wwyx(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.y(), self.x()])
    }

    #[inline(always)]
    pub const fn wwyy(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.y(), self.y()])
    }

    #[inline(always)]
    pub const fn wwyz(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.y(), self.z()])
    }

    #[inline(always)]
    pub const fn wwyw(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.y(), self.w()])
    }

    #[inline(always)]
    pub const fn wwzx(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.z(), self.x()])
    }

    #[inline(always)]
    pub const fn wwzy(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.z(), self.y()])
    }

    #[inline(always)]
    pub const fn wwzz(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.z(), self.z()])
    }

    #[inline(always)]
    pub const fn wwzw(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.z(), self.w()])
    }

    #[inline(always)]
    pub const fn wwwx(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.w(), self.x()])
    }

    #[inline(always)]
    pub const fn wwwy(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.w(), self.y()])
    }

    #[inline(always)]
    pub const fn wwwz(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.w(), self.z()])
    }

    #[inline(always)]
    pub const fn wwww(&self) -> Vec4 {
        Vec4([self.w(), self.w(), self.w(), self.w()])
    }

    /// Linearly interpolates between two `Vec4` values component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec4([
            lerp(a.x(), b.x(), t),
            lerp(a.y(), b.y(), t),
            lerp(a.z(), b.z(), t),
            lerp(a.w(), b.w(), t),
        ])
    }
}

/// A region in 3D space defined by a points on a diagonal.
#[derive(Clone, Copy)]
pub struct Region3 {
    /// The corner with the smallest component values.
    pub min: Vec3,
    /// The corner with the largest component values.
    pub max: Vec3,
}

impl Region3 {
    /// Builds the axis-aligned bounding box enclosing all `points`.
    pub fn new(points: impl Iterator<Item = Vec3>) -> Self {
        let mut min = Vec3([f32::INFINITY; 3]);
        let mut max = Vec3([f32::NEG_INFINITY; 3]);

        for point in points {
            for i in 0..3 {
                min.0[i] = min.0[i].min(point.0[i]);
                max.0[i] = max.0[i].max(point.0[i]);
            }
        }

        Region3 { min, max }
    }

    /// Returns the minimum corner.
    pub fn min(&self) -> Vec3 {
        self.min
    }

    /// Returns the maximum corner.
    pub fn max(&self) -> Vec3 {
        self.max
    }

    /// Returns `true` if any axis has min > max (degenerate region).
    pub fn is_empty(&self) -> bool {
        self.min.x() > self.max.x() || self.min.y() > self.max.y() || self.min.z() > self.max.z()
    }

    /// Returns `true` if min equals max (a single point).
    pub fn is_singular(&self) -> bool {
        self.min == self.max
    }

    /// Returns the center point of the region.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns 4 diagonals of the region.
    pub fn diagonals(&self) -> [(Vec3, Vec3); 4] {
        [
            (self.min, self.max),
            (
                Vec3([self.min.x(), self.min.y(), self.max.z()]),
                Vec3([self.max.x(), self.max.y(), self.min.z()]),
            ),
            (
                Vec3([self.min.x(), self.max.y(), self.min.z()]),
                Vec3([self.max.x(), self.min.y(), self.max.z()]),
            ),
            (
                Vec3([self.max.x(), self.min.y(), self.min.z()]),
                Vec3([self.min.x(), self.max.y(), self.max.z()]),
            ),
        ]
    }

    /// Returns 4 normalized diagonal axes of the region.
    pub fn diagonal_axes(&self) -> [Vec3; 4] {
        [
            Vec3([
                self.max.x() - self.min.x(),
                self.max.y() - self.min.y(),
                self.max.z() - self.min.z(),
            ])
            .norm(),
            Vec3([
                self.max.x() - self.min.x(),
                self.max.y() - self.min.y(),
                self.min.z() - self.max.z(),
            ])
            .norm(),
            Vec3([
                self.max.x() - self.min.x(),
                self.min.y() - self.max.y(),
                self.max.z() - self.min.z(),
            ])
            .norm(),
            Vec3([
                self.min.x() - self.max.x(),
                self.max.y() - self.min.y(),
                self.max.z() - self.min.z(),
            ])
            .norm(),
        ]
    }

    /// Returns `true` if the region is non-degenerate (min ≤ max on every axis).
    pub fn is_real(&self) -> bool {
        self.min.x() <= self.max.x() && self.min.y() <= self.max.y() && self.min.z() <= self.max.z()
    }

    /// Returns the volume of the region, or `0.0` if degenerate.
    pub fn volume(&self) -> f32 {
        let diff = self.max - self.min;
        diff.x().min(0.0) * diff.y().min(0.0) * diff.z().min(0.0)
    }
}

/// A single-channel 8-bit unsigned color (red).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct R8U(u8);

impl_fixedcode_struct!(R8U(r: u8) | Infallible);

impl R8U {
    /// Full white (maximum value).
    pub const WHITE: R8U = R8U(255);
    /// Full black (zero value).
    pub const BLACK: R8U = R8U(0);

    /// Creates a new single-channel color from a `u8` value.
    #[inline(always)]
    pub const fn new(r: u8) -> Self {
        R8U(r)
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn bytes(&self) -> [u8; 1] {
        [self.0]
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 1]) -> Self {
        R8U(bytes[0])
    }

    /// Returns the raw `u8` bit pattern.
    #[inline(always)]
    pub const fn bits(&self) -> u8 {
        self.0
    }

    /// Constructs from a raw `u8` bit pattern.
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> u8 {
        self.0
    }

    /// Converts this color to its 32-bit float representation.
    #[inline(always)]
    pub const fn into_f32(self) -> R32F {
        R32F(self.0 as f32 / 255.0)
    }

    /// Converts from a 32-bit float representation, clamping to [0, 255].
    #[inline(always)]
    pub const fn from_f32(luma: R32F) -> R8U {
        let clamped = (luma.r() * 255.0).clamp(0.0, 255.0);
        R8U(clamped as u8)
    }

    /// Wrapping unsigned addition per channel.
    #[inline(always)]
    pub fn wrapping_add(self, other: Self) -> Self {
        R8U(self.0.wrapping_add(other.0))
    }

    /// Wrapping unsigned subtraction per channel.
    #[inline(always)]
    pub fn wrapping_sub(self, other: Self) -> Self {
        R8U(self.0.wrapping_sub(other.0))
    }

    /// Returns the per-channel difference.
    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> f32 {
        a.r() as f32 - b.r() as f32
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff * diff
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::diff(a, b)
    }
}

/// A single-channel color represented as one `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct R32F(f32);

impl_fixedcode_struct!(R32F(r: f32) | Infallible);

impl R32F {
    /// Full white (maximum value).
    pub const WHITE: R32F = R32F(1.0);
    /// Full black (zero value).
    pub const BLACK: R32F = R32F(0.0);

    /// Creates a new single-channel color from an `f32` value.
    #[inline(always)]
    pub const fn new(r: f32) -> Self {
        R32F(r)
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0
    }

    /// Extends to a two-channel color by appending a green component.
    #[inline(always)]
    pub const fn with_g(self, g: f32) -> Rg32F {
        Rg32F([self.r(), g])
    }

    /// Linearly interpolates between two colors component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        R32F(lerp(a.r(), b.r(), t))
    }

    /// Returns the per-channel difference.
    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> f32 {
        a.r() - b.r()
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff * diff
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::diff(a, b)
    }

    /// Translates this color by the given offset.
    #[inline(always)]
    pub const fn offset(self, offset: f32) -> Self {
        R32F(self.r() + offset)
    }
}

/// A two-channel 8-bit unsigned color (red, green).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rg8U([u8; 2]);

impl_fixedcode_array!(Rg8U([u8; 2]) | Infallible);

impl Rg8U {
    /// Full white (all channels maximum).
    pub const WHITE: Rg8U = Rg8U([255, 255]);
    /// Full black (all channels zero).
    pub const BLACK: Rg8U = Rg8U([0, 0]);

    /// Creates a new two-channel color from `u8` red and green values.
    #[inline(always)]
    pub const fn new(r: u8, g: u8) -> Self {
        Rg8U([r, g])
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn bytes(&self) -> [u8; 2] {
        self.0
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Rg8U(bytes)
    }

    /// Returns the raw `u16` bit pattern (little-endian).
    #[inline(always)]
    pub const fn bits(&self) -> u16 {
        u16::from_le_bytes(self.0)
    }

    /// Constructs from a raw `u16` bit pattern (little-endian).
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Rg8U(bits.to_le_bytes())
    }

    /// Returns the channels as an interleaved `u16` bit pattern.
    #[inline(always)]
    pub const fn bits_interleaved(&self) -> u16 {
        let [r, g] = self.0;
        interleave8_2(r, g)
    }

    /// Constructs from an interleaved `u16` bit pattern.
    #[inline(always)]
    pub const fn from_bits_interleaved(bits: u16) -> Self {
        let (r, g) = deinterleave8_2(bits);
        Rg8U::new(r, g)
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> u8 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> u8 {
        self.0[1]
    }

    /// Converts this color to its 32-bit float representation.
    #[inline(always)]
    pub const fn into_f32(self) -> Rg32F {
        Rg32F([self.r() as f32 / 255.0, self.g() as f32 / 255.0])
    }

    /// Converts from a 32-bit float representation, clamping each channel to [0, 255].
    #[inline(always)]
    pub const fn from_f32(rg: Rg32F) -> Rg8U {
        let r = (rg.r() * 255.0).clamp(0.0, 255.0);
        let g = (rg.g() * 255.0).clamp(0.0, 255.0);
        Rg8U([r as u8, g as u8])
    }

    /// Wrapping unsigned addition per channel.
    #[inline(always)]
    pub fn wrapping_add(self, other: Self) -> Self {
        Rg8U([
            self.r().wrapping_add(other.r()),
            self.g().wrapping_add(other.g()),
        ])
    }

    /// Wrapping unsigned subtraction per channel.
    #[inline(always)]
    pub fn wrapping_sub(self, other: Self) -> Self {
        Rg8U([
            self.r().wrapping_sub(other.r()),
            self.g().wrapping_sub(other.g()),
        ])
    }

    /// Returns the per-channel difference as a `Vec2`.
    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> Vec2 {
        Vec2([a.r() as f32 - b.r() as f32, a.g() as f32 - b.g() as f32])
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff.dot(diff)
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }
}

/// A two-channel (red, green) color represented as 2 `f32`s.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Rg32F([f32; 2]);

impl_fixedcode_array!(Rg32F([f32; 2]) | Infallible);

impl Rg32F {
    /// Full white (all channels maximum).
    pub const WHITE: Rg32F = Rg32F([1.0, 1.0]);
    /// Full black (all channels zero).
    pub const BLACK: Rg32F = Rg32F([0.0, 0.0]);

    /// Creates a new two-channel color from `f32` red and green values.
    #[inline(always)]
    pub const fn new(r: f32, g: f32) -> Self {
        Rg32F([r, g])
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> f32 {
        self.0[1]
    }

    /// Extends to a three-channel color by appending a blue component.
    #[inline(always)]
    pub const fn with_b(self, b: f32) -> Rgb32F {
        Rgb32F([self.r(), self.g(), b])
    }

    /// Linearly interpolates between two colors component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rg32F([lerp(a.r(), b.r(), t), lerp(a.g(), b.g(), t)])
    }

    /// Returns the per-channel difference as a `Vec2`.
    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> Vec2 {
        Vec2([a.r() - b.r(), a.g() - b.g()])
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff.dot(diff)
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }

    /// Translates this color by the given offset.
    #[inline(always)]
    pub const fn offset(self, offset: Vec2) -> Self {
        Rg32F([self.r() + offset.x(), self.g() + offset.y()])
    }
}

/// An RGB color with 8 bit unsigned normalized integers per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rgb8U([u8; 3]);

impl_fixedcode_array!(Rgb8U([u8; 3]) | Infallible);

impl Rgb8U {
    /// Full white (all channels maximum).
    pub const WHITE: Rgb8U = Rgb8U([255, 255, 255]);
    /// Full black (all channels zero).
    pub const BLACK: Rgb8U = Rgb8U([0, 0, 0]);
    /// Pure red.
    pub const RED: Rgb8U = Rgb8U([255, 0, 0]);
    /// Pure green.
    pub const GREEN: Rgb8U = Rgb8U([0, 255, 0]);
    /// Pure blue.
    pub const BLUE: Rgb8U = Rgb8U([0, 0, 255]);

    /// Creates a new three-channel color from `u8` red, green, and blue values.
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Rgb8U([r, g, b])
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn bytes(&self) -> [u8; 3] {
        self.0
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 3]) -> Self {
        Rgb8U(bytes)
    }

    /// Returns the raw `u32` bit pattern (little-endian, high byte zero).
    #[inline(always)]
    pub const fn bits(&self) -> u32 {
        let [r, g, b] = self.0;
        u32::from_le_bytes([r, g, b, 0])
    }

    /// Constructs from a raw `u32` bit pattern (little-endian, high byte ignored).
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        let [r, g, b, _] = bits.to_le_bytes();
        Rgb8U([r, g, b])
    }

    /// Returns the channels as an interleaved `u32` bit pattern.
    #[inline(always)]
    pub const fn bits_interleaved(&self) -> u32 {
        let [r, g, b] = self.0;
        interleave8_3(r, b, g)
    }

    /// Constructs from an interleaved `u32` bit pattern.
    #[inline(always)]
    pub const fn from_bits_interleaved(bits: u32) -> Self {
        let (r, b, g) = deinterleave8_3(bits);
        Rgb8U::new(r, g, b)
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> u8 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> u8 {
        self.0[1]
    }

    /// Returns the blue channel value.
    #[inline(always)]
    pub const fn b(&self) -> u8 {
        self.0[2]
    }

    /// Sets the red channel value.
    #[inline(always)]
    pub fn set_r(&mut self, r: u8) {
        self.0[0] = r;
    }

    /// Sets the green channel value.
    #[inline(always)]
    pub fn set_g(&mut self, g: u8) {
        self.0[1] = g;
    }

    /// Sets the blue channel value.
    #[inline(always)]
    pub fn set_b(&mut self, b: u8) {
        self.0[2] = b;
    }

    /// Extends to a four-channel color by appending an alpha component.
    #[inline(always)]
    pub const fn with_alpha(self, a: u8) -> Rgba8U {
        Rgba8U([self.r(), self.g(), self.b(), a])
    }

    /// Extends to a four-channel color with full opacity (alpha = 255).
    #[inline(always)]
    pub const fn into_opaque(self) -> Rgba8U {
        Rgba8U([self.r(), self.g(), self.b(), 255])
    }

    /// Converts this color to its 32-bit float representation.
    #[inline(always)]
    pub const fn into_f32(self) -> Rgb32F {
        let [r, g, b] = self.0;
        Rgb32F([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
    }

    /// Converts from a 32-bit float representation, clamping each channel to [0, 255].
    #[inline(always)]
    pub fn from_f32(rgb: Rgb32F) -> Self {
        let r = (rgb.r() * 255.0).clamp(0.0, 255.0) as u8;
        let g = (rgb.g() * 255.0).clamp(0.0, 255.0) as u8;
        let b = (rgb.b() * 255.0).clamp(0.0, 255.0) as u8;
        Rgb8U::new(r, g, b)
    }

    /// Wrapping unsigned addition per channel.
    #[inline(always)]
    pub fn wrapping_add(lhs: Self, rhs: Self) -> Self {
        let r = lhs.r().wrapping_add(rhs.r());
        let g = lhs.g().wrapping_add(rhs.g());
        let b = lhs.b().wrapping_add(rhs.b());
        Rgb8U::new(r, g, b)
    }

    /// Wrapping unsigned subtraction per channel.
    #[inline(always)]
    pub fn wrapping_sub(lhs: Self, rhs: Self) -> Self {
        let r = lhs.r().wrapping_sub(rhs.r());
        let g = lhs.g().wrapping_sub(rhs.g());
        let b = lhs.b().wrapping_sub(rhs.b());
        Rgb8U::new(r, g, b)
    }

    /// Returns the per-channel difference as a `Vec3`.
    #[inline(always)]
    pub const fn diff(lhs: Self, rhs: Self) -> Vec3 {
        Vec3([
            lhs.r() as f32 - rhs.r() as f32,
            lhs.g() as f32 - rhs.g() as f32,
            lhs.b() as f32 - rhs.b() as f32,
        ])
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(lhs: Self, rhs: Self) -> f32 {
        let diff = Self::diff(lhs, rhs);
        diff.dot(diff)
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(lhs: Self, rhs: Self) -> f32 {
        Self::distance_squared(lhs, rhs).sqrt()
    }
}

/// An RGB color represented as 3 floats.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Rgb32F([f32; 3]);

impl_fixedcode_array!(Rgb32F([f32; 3]) | Infallible);

impl Rgb32F {
    /// Full white (all channels maximum).
    pub const WHITE: Rgb32F = Rgb32F([1.0, 1.0, 1.0]);
    /// Full black (all channels zero).
    pub const BLACK: Rgb32F = Rgb32F([0.0, 0.0, 0.0]);
    /// Pure red.
    pub const RED: Rgb32F = Rgb32F([1.0, 0.0, 0.0]);
    /// Pure green.
    pub const GREEN: Rgb32F = Rgb32F([0.0, 1.0, 0.0]);
    /// Pure blue.
    pub const BLUE: Rgb32F = Rgb32F([0.0, 0.0, 1.0]);

    /// Creates a new three-channel color from `f32` red, green, and blue values.
    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Rgb32F([r, g, b])
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> f32 {
        self.0[1]
    }

    /// Returns the blue channel value.
    #[inline(always)]
    pub const fn b(&self) -> f32 {
        self.0[2]
    }

    /// Extends to a four-channel color by appending an alpha component.
    #[inline(always)]
    pub const fn with_alpha(self, a: f32) -> Rgba32F {
        Rgba32F([self.r(), self.g(), self.b(), a])
    }

    /// Extends to a four-channel color with full opacity (alpha = 1.0).
    #[inline(always)]
    pub const fn into_opaque(self) -> Rgba32F {
        Rgba32F([self.r(), self.g(), self.b(), 1.0])
    }

    /// Linearly interpolates between two colors component-wise.
    #[inline(always)]
    pub fn lerp(lhs: Self, rhs: Self, t: f32) -> Self {
        Rgb32F([
            lerp(lhs.r(), rhs.r(), t),
            lerp(lhs.g(), rhs.g(), t),
            lerp(lhs.b(), rhs.b(), t),
        ])
    }

    /// Returns the per-channel difference as a `Vec3`.
    #[inline(always)]
    pub const fn diff(lhs: Self, rhs: Self) -> Vec3 {
        Vec3([lhs.r() - rhs.r(), lhs.g() - rhs.g(), lhs.b() - rhs.b()])
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(lhs: Self, rhs: Self) -> f32 {
        let diff = Self::diff(lhs, rhs);
        diff.dot(diff)
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(lhs: Self, rhs: Self) -> f32 {
        Self::distance_squared(lhs, rhs).sqrt()
    }

    /// Translates this color by the given offset.
    #[inline(always)]
    pub const fn offset(self, offset: Vec3) -> Self {
        Rgb32F([
            self.r() + offset.x(),
            self.g() + offset.y(),
            self.b() + offset.z(),
        ])
    }
}

/// An RGBA color with 8 bit unsigned normalized integers per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rgba8U([u8; 4]);

impl_fixedcode_array!(Rgba8U([u8; 4]) | Infallible);

impl Rgba8U {
    /// Full white (all channels maximum).
    pub const WHITE: Rgba8U = Rgba8U([255, 255, 255, 255]);
    /// Full black (all channels zero, fully opaque).
    pub const BLACK: Rgba8U = Rgba8U([0, 0, 0, 255]);
    /// Pure red.
    pub const RED: Rgba8U = Rgba8U([255, 0, 0, 255]);
    /// Pure green.
    pub const GREEN: Rgba8U = Rgba8U([0, 255, 0, 255]);
    /// Pure blue.
    pub const BLUE: Rgba8U = Rgba8U([0, 0, 255, 255]);
    /// Fully transparent black.
    pub const TRANSPARENT: Rgba8U = Rgba8U([0, 0, 0, 0]);

    /// Creates a new four-channel color from `u8` red, green, blue, and alpha values.
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Rgba8U([r, g, b, a])
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn bytes(&self) -> [u8; 4] {
        self.0
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Rgba8U(bytes)
    }

    /// Returns the raw `u32` bit pattern (little-endian).
    #[inline(always)]
    pub const fn bits(&self) -> u32 {
        u32::from_le_bytes(self.0)
    }

    /// Constructs from a raw `u32` bit pattern (little-endian).
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        let [r, g, b, a] = bits.to_le_bytes();
        Rgba8U([r, g, b, a])
    }

    /// Returns the channels as an interleaved `u32` bit pattern.
    #[inline(always)]
    pub const fn bits_interleaved(&self) -> u32 {
        let [r, g, b, a] = self.0;
        interleave8_4(r, b, g, a)
    }

    /// Constructs from an interleaved `u32` bit pattern.
    #[inline(always)]
    pub const fn from_bits_interleaved(bits: u32) -> Self {
        let (r, b, g, a) = deinterleave8_4(bits);
        Rgba8U::new(r, g, b, a)
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> u8 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> u8 {
        self.0[1]
    }

    /// Returns the blue channel value.
    #[inline(always)]
    pub const fn b(&self) -> u8 {
        self.0[2]
    }

    /// Returns the alpha channel value.
    #[inline(always)]
    pub const fn a(&self) -> u8 {
        self.0[3]
    }

    /// Sets the red channel value.
    #[inline(always)]
    pub fn set_r(&mut self, r: u8) {
        self.0[0] = r;
    }

    /// Sets the green channel value.
    #[inline(always)]
    pub fn set_g(&mut self, g: u8) {
        self.0[1] = g;
    }

    /// Sets the blue channel value.
    #[inline(always)]
    pub fn set_b(&mut self, b: u8) {
        self.0[2] = b;
    }

    /// Sets the alpha channel value.
    #[inline(always)]
    pub fn set_a(&mut self, a: u8) {
        self.0[3] = a;
    }

    /// Converts this color to its 32-bit float representation.
    #[inline(always)]
    pub const fn into_f32(self) -> Rgba32F {
        let [r, g, b, a] = self.0;
        Rgba32F([
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ])
    }

    /// Converts from a 32-bit float representation, clamping each channel to [0, 255].
    #[inline(always)]
    pub fn from_f32(rgb: Rgba32F) -> Self {
        let r = (rgb.r() * 255.0).clamp(0.0, 255.0) as u8;
        let g = (rgb.g() * 255.0).clamp(0.0, 255.0) as u8;
        let b = (rgb.b() * 255.0).clamp(0.0, 255.0) as u8;
        let a = (rgb.a() * 255.0).clamp(0.0, 255.0) as u8;
        Rgba8U::new(r, g, b, a)
    }

    /// Returns only the RGB channels, discarding alpha.
    #[inline(always)]
    pub const fn rgb(&self) -> Rgb8U {
        Rgb8U::new(self.r(), self.g(), self.b())
    }

    /// Wrapping unsigned addition per channel.
    #[inline(always)]
    pub fn wrapping_add(lhs: Self, rhs: Self) -> Self {
        let r = lhs.r().wrapping_add(rhs.r());
        let g = lhs.g().wrapping_add(rhs.g());
        let b = lhs.b().wrapping_add(rhs.b());
        let a = lhs.a().wrapping_add(rhs.a());
        Rgba8U::new(r, g, b, a)
    }

    /// Wrapping unsigned subtraction per channel.
    #[inline(always)]
    pub fn wrapping_sub(lhs: Self, rhs: Self) -> Self {
        let r = lhs.r().wrapping_sub(rhs.r());
        let g = lhs.g().wrapping_sub(rhs.g());
        let b = lhs.b().wrapping_sub(rhs.b());
        let a = lhs.a().wrapping_sub(rhs.a());

        Rgba8U::new(r, g, b, a)
    }
}

/// An RGBA color represented as 4 `f32`s.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Rgba32F([f32; 4]);

impl_fixedcode_array!(Rgba32F([f32; 4]) | Infallible);

impl Rgba32F {
    /// Full white (all channels maximum).
    pub const WHITE: Rgba32F = Rgba32F([1.0, 1.0, 1.0, 1.0]);
    /// Full black (all channels zero, fully opaque).
    pub const BLACK: Rgba32F = Rgba32F([0.0, 0.0, 0.0, 1.0]);
    /// Pure red.
    pub const RED: Rgba32F = Rgba32F([1.0, 0.0, 0.0, 1.0]);
    /// Pure green.
    pub const GREEN: Rgba32F = Rgba32F([0.0, 1.0, 0.0, 1.0]);
    /// Pure blue.
    pub const BLUE: Rgba32F = Rgba32F([0.0, 0.0, 1.0, 1.0]);
    /// Fully transparent black.
    pub const TRANSPARENT: Rgba32F = Rgba32F([0.0, 0.0, 0.0, 0.0]);

    /// Creates a new four-channel color from `f32` red, green, blue, and alpha values.
    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Rgba32F([r, g, b, a])
    }

    /// Returns the red channel value.
    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0[0]
    }

    /// Returns the green channel value.
    #[inline(always)]
    pub const fn g(&self) -> f32 {
        self.0[1]
    }

    /// Returns the blue channel value.
    #[inline(always)]
    pub const fn b(&self) -> f32 {
        self.0[2]
    }

    /// Returns the alpha channel value.
    #[inline(always)]
    pub const fn a(&self) -> f32 {
        self.0[3]
    }

    /// Returns only the RGB channels, discarding alpha.
    #[inline(always)]
    pub const fn rgb(&self) -> Rgb32F {
        Rgb32F([self.r(), self.g(), self.b()])
    }

    /// Linearly interpolates between two colors component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgba32F([
            lerp(a.r(), b.r(), t),
            lerp(a.g(), b.g(), t),
            lerp(a.b(), b.b(), t),
            lerp(a.a(), b.a(), t),
        ])
    }
}

/// An RGB color with 5,6 and 5 bits unsigned normalized integers per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rgb565(u16);

impl_fixedcode_struct!(Rgb565(rgb: u16) | Infallible);

#[allow(clippy::unusual_byte_groupings)] // Grouped by RGB565 bit-field widths (5-6-5).
impl Rgb565 {
    /// Full white (all channels maximum).
    pub const WHITE: Rgb565 = Rgb565(0b11111_111111_11111);
    /// Full black (all channels zero).
    pub const BLACK: Rgb565 = Rgb565(0);

    /// Creates a new color from 5-bit red, 6-bit green, and 5-bit blue values.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `r` ≤ 31, `g` ≤ 63, and `b` ≤ 31.
    #[inline(always)]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        debug_assert!(r <= 31, "Red channel must be in range 0..=31");
        debug_assert!(g <= 63, "Green channel must be in range 0..=63");
        debug_assert!(b <= 31, "Blue channel must be in range 0..=31");

        let r = (r & 0x1F) as u16;
        let g = (g & 0x3F) as u16;
        let b = (b & 0x1F) as u16;
        Rgb565((r << 11) | (g << 5) | b)
    }

    /// Return the raw bits of the encoded color.
    #[inline(always)]
    pub const fn bits(&self) -> u16 {
        self.0
    }

    /// Return color from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Rgb565(bits)
    }

    /// Returns the channels as an interleaved `u16` bit pattern.
    #[inline(always)]
    pub const fn bits_interleaved(&self) -> u16 {
        let r = self.r();
        let g = self.g();
        let b = self.b();

        interleave655_3(g, r, b)
    }

    /// Constructs from an interleaved `u16` bit pattern.
    #[inline(always)]
    pub const fn from_bits_interleaved(bits: u16) -> Self {
        let (g, r, b) = deinterleave655_3(bits);
        Rgb565::new(r, g, b)
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn from_bytes(bytes: [u8; 2]) -> Self {
        Rgb565(u16::from_le_bytes(bytes))
    }

    /// Return color from raw bytes.
    #[inline(always)]
    pub const fn bytes(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Returns the red channel value (5 bits).
    #[inline(always)]
    pub const fn r(&self) -> u8 {
        (self.0 >> 11) as u8
    }

    /// Returns the green channel value (6 bits).
    #[inline(always)]
    pub const fn g(&self) -> u8 {
        ((self.0 >> 5) & 0b111111) as u8
    }

    /// Returns the blue channel value (5 bits).
    #[inline(always)]
    pub const fn b(&self) -> u8 {
        (self.0 & 0b11111) as u8
    }

    /// Sets the red channel value (5 bits).
    #[inline(always)]
    pub fn set_r(&mut self, r: u8) {
        debug_assert!(r <= 31, "Red channel must be in range 0..=31");
        self.0 = (self.0 & 0b00000_111111_11111) | ((r as u16) << 11);
    }

    /// Sets the green channel value (6 bits).
    #[inline(always)]
    pub fn set_g(&mut self, g: u8) {
        debug_assert!(g <= 63, "Green channel must be in range 0..=63");
        self.0 = (self.0 & 0b11111_000000_11111) | ((g as u16) << 5);
    }

    /// Sets the blue channel value (5 bits).
    #[inline(always)]
    pub fn set_b(&mut self, b: u8) {
        debug_assert!(b <= 31, "Blue channel must be in range 0..=31");
        self.0 = (self.0 & 0b11111_111111_00000) | (b as u16);
    }

    /// Converts this color to its `Rgb32F` float representation.
    #[inline(always)]
    pub const fn into_f32(self) -> Rgb32F {
        let r = ((self.0 >> 11) & 0b11111) as f32 / 31.0;
        let g = ((self.0 >> 5) & 0b111111) as f32 / 63.0;
        let b = (self.0 & 0b11111) as f32 / 31.0;
        Rgb32F([r, g, b])
    }

    /// Converts from an `Rgb32F` float representation, clamping to valid ranges.
    #[inline(always)]
    pub fn from_f32(rgb: Rgb32F) -> Self {
        let [r, g, b] = rgb.0;
        let r = (r * 31.0).clamp(0.0, 31.0) as u16;
        let g = (g * 63.0).clamp(0.0, 63.0) as u16;
        let b = (b * 31.0).clamp(0.0, 31.0) as u16;
        Rgb565((r << 11) | (g << 5) | b)
    }

    /// Wrapping unsigned addition per channel, masked to valid bit widths.
    #[inline(always)]
    pub fn wrapping_add(a: Self, b: Self) -> Self {
        let r = a.r().wrapping_add(b.r()) & 31;
        let g = a.g().wrapping_add(b.g()) & 63;
        let b = a.b().wrapping_add(b.b()) & 31;
        Rgb565::new(r, g, b)
    }

    /// Wrapping unsigned subtraction per channel, masked to valid bit widths.
    #[inline(always)]
    pub fn wrapping_sub(a: Self, b: Self) -> Self {
        let r = a.r().wrapping_sub(b.r()) & 31;
        let g = a.g().wrapping_sub(b.g()) & 63;
        let b = a.b().wrapping_sub(b.b()) & 31;
        Rgb565::new(r, g, b)
    }

    /// Converts to an `Rgb8U` color via float intermediary.
    #[inline(always)]
    pub fn into_8u(self) -> Rgb8U {
        Rgb8U::from_f32(self.into_f32())
    }

    /// Converts from an `Rgb8U` color via float intermediary.
    #[inline(always)]
    pub fn from_8u(rgb: Rgb8U) -> Self {
        Self::from_f32(rgb.into_f32())
    }
}

/// An YIQ color represented as 3 floats.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Yiq32F([f32; 3]);

impl_fixedcode_array!(Yiq32F([f32; 3]) | Infallible);

impl Yiq32F {
    /// Full white (luminance 1.0, no chrominance).
    pub const WHITE: Yiq32F = Yiq32F([1.0, 0.0, 0.0]);
    /// Full black (all components zero).
    pub const BLACK: Yiq32F = Yiq32F([0.0, 0.0, 0.0]);

    /// Creates a new YIQ color from luminance (`y`), in-phase (`i`), and quadrature (`q`).
    #[inline(always)]
    pub const fn new(y: f32, i: f32, q: f32) -> Self {
        Yiq32F([y, i, q])
    }

    /// Returns the luminance (Y) component.
    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[0]
    }

    /// Returns the in-phase (I) chrominance component.
    #[inline(always)]
    pub const fn i(&self) -> f32 {
        self.0[1]
    }

    /// Returns the quadrature (Q) chrominance component.
    #[inline(always)]
    pub const fn q(&self) -> f32 {
        self.0[2]
    }

    /// Converts an `Rgb32F` color to YIQ color space.
    #[inline(always)]
    pub const fn from_rgb(rgb: Rgb32F) -> Self {
        let [r, g, b] = rgb.0;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let i = 0.5959 * r - 0.2746 * g - 0.3213 * b;
        let q = 0.2115 * r - 0.5227 * g + 0.3112 * b;
        Yiq32F([y, i, q])
    }

    /// Converts this YIQ color back to RGB color space.
    #[inline(always)]
    pub const fn into_rgb(self) -> Rgb32F {
        let [y, i, q] = self.0;
        let r = y + 0.956 * i + 0.619 * q;
        let g = y - 0.272 * i - 0.647 * q;
        let b = y - 1.106 * i + 1.703 * q;
        Rgb32F([r, g, b])
    }

    /// Linearly interpolates between two colors component-wise.
    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Yiq32F([
            lerp(a.y(), b.y(), t),
            lerp(a.i(), b.i(), t),
            lerp(a.q(), b.q(), t),
        ])
    }

    /// Returns the perceptual distance, weighting luminance higher than chrominance.
    #[inline(always)]
    pub fn perceptual_distance(a: Self, b: Self) -> f32 {
        let [y1, i1, q1] = a.0;
        let [y2, i2, q2] = b.0;

        let luminance_diff = (y1 - y2) * (y1 - y2);
        let chrominance_diff = 0.25 * ((i1 - i2) * (i1 - i2) + (q1 - q2) * (q1 - q2));

        (luminance_diff + chrominance_diff).sqrt()
    }

    /// Returns the per-channel difference as a `Vec3`.
    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> Vec3 {
        Vec3([a.y() - b.y(), a.i() - b.i(), a.q() - b.q()])
    }

    /// Returns the squared Euclidean distance.
    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff.dot(diff)
    }

    /// Returns the Euclidean distance.
    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }

    /// Translates this color by the given offset.
    #[inline(always)]
    pub const fn offset(self, offset: Vec3) -> Self {
        Yiq32F([
            self.y() + offset.x(),
            self.i() + offset.y(),
            self.q() + offset.z(),
        ])
    }
}

impl From<Rgb32F> for Yiq32F {
    #[inline(always)]
    fn from(rgb: Rgb32F) -> Self {
        Yiq32F::from_rgb(rgb)
    }
}

impl From<Yiq32F> for Rgb32F {
    #[inline(always)]
    fn from(yiq: Yiq32F) -> Self {
        yiq.into_rgb()
    }
}

impl From<Rgb32F> for Vec3 {
    #[inline(always)]
    fn from(value: Rgb32F) -> Self {
        Vec3([value.r(), value.g(), value.b()])
    }
}

impl From<Yiq32F> for Vec3 {
    #[inline(always)]
    fn from(value: Yiq32F) -> Self {
        Vec3([value.y(), value.i(), value.q()])
    }
}

impl From<Rgba32F> for Vec4 {
    #[inline(always)]
    fn from(value: Rgba32F) -> Self {
        Vec4([value.r(), value.g(), value.b(), value.a()])
    }
}

impl From<Vec3> for Rgb32F {
    #[inline(always)]
    fn from(value: Vec3) -> Self {
        Rgb32F([value.x(), value.y(), value.z()])
    }
}

impl From<Vec3> for Yiq32F {
    #[inline(always)]
    fn from(value: Vec3) -> Self {
        Yiq32F([value.x(), value.y(), value.z()])
    }
}

impl From<Vec4> for Rgba32F {
    #[inline(always)]
    fn from(value: Vec4) -> Self {
        Rgba32F([value.x(), value.y(), value.z(), value.w()])
    }
}

/// Returns the bounding-box diagonal axis along which the given samples
/// have the greatest variance.
pub fn max_variance_diagonal_axis(samples: &[Vec3]) -> Vec3 {
    let region = Region3::new(samples.iter().copied());
    let center = region.center();
    let diagonals = region.diagonal_axes();

    let mut best_diagonal = Vec3::ZERO;
    let mut best_var = -1.0f32;

    for &diagonal in &diagonals[0..] {
        let mut var = 0.0f32;
        for &v in samples {
            let t = (v - center).dot(diagonal);
            var += t * t;
        }
        if var > best_var {
            best_var = var;
            best_diagonal = diagonal;
        }
    }

    best_diagonal
}

/// Estimates the principal component axis of a set of 3D points via
/// power iteration on the covariance matrix.
pub fn pca_axis(v: &[Vec3]) -> Vec3 {
    #![allow(clippy::needless_range_loop)]
    let n = v.len() as f32;
    let mut mean = Vec3::ZERO;
    for p in v {
        mean += *p;
    }
    mean /= n;

    let mut cov = [[0.0; 3]; 3];
    for p in v {
        let d = *p - mean;
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d.0[i] * d.0[j];
            }
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= n;
        }
    }

    let diagonal = max_variance_diagonal_axis(v);

    // Power iteration to find the principal component
    let mut axis = diagonal;
    for _ in 0..10 {
        let mut next_axis = Vec3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                next_axis.0[i] += cov[i][j] * axis.0[j];
            }
        }

        let len = next_axis.length();
        if len > 1.0e-6 {
            next_axis /= len;
        } else {
            next_axis = diagonal;
        }
        axis = next_axis;
    }

    axis
}

/// An axis-aligned rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect<T> {
    /// Horizontal origin.
    pub x: T,
    /// Vertical origin.
    pub y: T,
    /// Width.
    pub w: T,
    /// Height.
    pub h: T,
}

impl<T> Rect<T> {
    /// Returns the area of the rectangle.
    pub fn area(&self) -> T
    where
        for<'a> &'a T: Mul<&'a T, Output = T>,
    {
        &self.w * &self.h
    }

    /// Returns `true` if the point (`x`, `y`) lies inside this rectangle.
    pub fn contains_point(&self, x: T, y: T) -> bool
    where
        for<'a> &'a T: Add<&'a T, Output = T>,
        T: PartialOrd,
    {
        self.x <= x && (&self.x + &self.w) > x && self.y <= y && (&self.y + &self.h) > y
    }

    /// Returns `true` if `other` is fully contained within this rectangle.
    pub fn contains(&self, other: &Rect<T>) -> bool
    where
        for<'a> &'a T: Add<&'a T, Output = T>,
        T: PartialOrd,
    {
        self.x <= other.x
            && self.y <= other.y
            && (&self.x + &self.w) >= (&other.x + &other.w)
            && (&self.y + &self.h) >= (&other.y + &other.h)
    }

    /// Shrinks the rectangle inward so that position and size are multiples of (`x`, `y`).
    pub fn round_in(&self, x: T, y: T) -> Self
    where
        T: Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Zero + Copy,
    {
        let ux = round_up(self.x, x);
        let uy = round_up(self.y, y);
        let uw = round_down(self.x + self.w, x) - ux;
        let uh = round_down(self.y + self.h, y) - uy;

        Rect {
            x: ux,
            y: uy,
            w: uw,
            h: uh,
        }
    }

    /// Grows the rectangle outward so that position and size are multiples of (`x`, `y`).
    pub fn round_out(&self, x: T, y: T) -> Self
    where
        T: Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Zero + Copy,
    {
        let ux = round_down(self.x, x);
        let uy = round_down(self.y, y);
        let uw = round_up(self.x + self.w, x) - ux;
        let uh = round_up(self.y + self.h, y) - uy;

        Rect {
            x: ux,
            y: uy,
            w: uw,
            h: uh,
        }
    }
}

/// Rounds `value` down to the nearest multiple of `round`.
#[inline(always)]
pub fn round_down<T>(value: T, round: T) -> T
where
    T: Sub<Output = T> + Rem<Output = T> + Copy,
{
    let rem = value % round;
    value - rem
}

/// Rounds `value` up to the nearest multiple of `round`.
#[inline(always)]
pub fn round_up<T>(value: T, round: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Rem<Output = T> + Zero + Copy,
{
    let rem = value % round;
    if rem.is_zero() {
        value
    } else {
        value + (round - rem)
    }
}

#[test]
fn test_round_down() {
    assert_eq!(7, round_down(13, 7));
    assert_eq!(14, round_down(14, 7));
    assert_eq!(14, round_down(15, 7));
}

#[test]
fn test_round_up() {
    assert_eq!(14, round_up(13, 7));
    assert_eq!(14, round_up(14, 7));
    assert_eq!(21, round_up(15, 7));
}

/// Spreads 32 bits into even bit positions of a 64-bit value (2-way interleave building block).
#[inline(always)]
pub const fn spread32_2(x: u32) -> u64 {
    let mut x = x as u64;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    x
}

/// Interleaves the bits of two 32-bit values into a 64-bit Morton code.
#[inline(always)]
pub const fn interleave32_2(x: u32, y: u32) -> u64 {
    spread32_2(x) | (spread32_2(y) << 1)
}

/// Extracts even bit positions from a 64-bit value back into 32 bits.
#[inline(always)]
pub const fn compact32_2(x: u64) -> u32 {
    let mut x = x;
    x = x & 0x5555555555555555;
    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
    x = x | (x >> 16);
    x as u32
}

/// Deinterleaves a 64-bit Morton code into two 32-bit values.
#[inline(always)]
pub const fn deinterleave32_2(x: u64) -> (u32, u32) {
    (compact32_2(x), compact32_2(x >> 1))
}

/// Spreads 16 bits into even bit positions of a 32-bit value.
#[inline(always)]
pub const fn spread16_2(x: u16) -> u32 {
    let mut x = x as u32;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Interleaves the bits of two 16-bit values into a 32-bit Morton code.
#[inline(always)]
pub const fn interleave16_2(x: u16, y: u16) -> u32 {
    spread16_2(x) | (spread16_2(y) << 1)
}

/// Extracts even bit positions from a 32-bit value back into 16 bits.
#[inline(always)]
pub const fn compact16_2(x: u32) -> u16 {
    let mut x = x;
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = x | (x >> 8);
    x as u16
}

/// Deinterleaves a 32-bit Morton code into two 16-bit values.
#[inline(always)]
pub const fn deinterleave16_2(x: u32) -> (u16, u16) {
    (compact16_2(x), compact16_2(x >> 1))
}

/// Spreads 8 bits into even bit positions of a 16-bit value.
#[inline(always)]
pub const fn spread8_2(x: u8) -> u16 {
    let mut x = x as u16;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;
    x
}

/// Interleaves the bits of two 8-bit values into a 16-bit Morton code.
#[inline(always)]
pub const fn interleave8_2(x: u8, y: u8) -> u16 {
    spread8_2(x) | (spread8_2(y) << 1)
}

/// Extracts even bit positions from a 16-bit value back into 8 bits.
#[inline(always)]
pub const fn compact8_2(x: u16) -> u8 {
    let mut x = x;
    x = x & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0F0F;
    x = x | (x >> 4);
    x as u8
}

/// Deinterleaves a 16-bit Morton code into two 8-bit values.
#[inline(always)]
pub const fn deinterleave8_2(x: u16) -> (u8, u8) {
    (compact8_2(x), compact8_2(x >> 1))
}

/// Spreads 8 bits into every-third bit position of a 24-bit value.
#[inline(always)]
pub const fn spread8_3(x: u8) -> u32 {
    let mut x = x as u32;
    x = (x | (x << 8)) & 0x00F00F;
    x = (x | (x << 4)) & 0x0C30C3;
    x = (x | (x << 2)) & 0x249249;
    x
}

/// Interleaves the bits of three 8-bit values into a 24-bit Morton code.
#[inline(always)]
pub const fn interleave8_3(x: u8, y: u8, z: u8) -> u32 {
    spread8_3(x) | (spread8_3(y) << 1) | (spread8_3(z) << 2)
}

/// Extracts every-third bit from a value back into 8 bits.
#[inline(always)]
pub const fn compact8_3(x: u32) -> u8 {
    let mut x = x;
    x = x & 0x249249;
    x = (x | (x >> 2)) & 0x0C30C3;
    x = (x | (x >> 4)) & 0x00F00F;
    x = (x | (x >> 8)) & 0x00000FFF;
    x as u8
}

/// Deinterleaves a 24-bit Morton code into three 8-bit values.
#[inline(always)]
pub const fn deinterleave8_3(x: u32) -> (u8, u8, u8) {
    (compact8_3(x), compact8_3(x >> 1), compact8_3(x >> 2))
}

/// Spreads 6 bits into every-third bit position of a 16-bit value.
#[inline(always)]
pub const fn spread6_3(x: u8) -> u16 {
    let mut x = x as u16;
    x = (x | (x << 8)) & 0x300F;
    x = (x | (x << 4)) & 0x30C3;
    x = (x | (x << 2)) & 0x9249;
    x
}

/// Interleaves three values (6, 5, 5 bits) into a 16-bit Morton code.
#[inline(always)]
pub const fn interleave655_3(x: u8, y: u8, z: u8) -> u16 {
    spread6_3(x) | (spread6_3(y) << 1) | (spread6_3(z) << 2)
}

/// Extracts every-third bit from a 16-bit value back into 6 bits.
#[inline(always)]
pub const fn compact6_3(x: u16) -> u8 {
    let mut x = x;
    x = x & 0x9249;
    x = (x | (x >> 2)) & 0x30C3;
    x = (x | (x >> 4)) & 0xF00F;
    x = (x | (x >> 8)) & 0x003F;
    x as u8
}

/// Deinterleaves a 16-bit Morton code into three values (6, 5, 5 bits).
#[inline(always)]
pub const fn deinterleave655_3(x: u16) -> (u8, u8, u8) {
    (compact6_3(x), compact6_3(x >> 1), compact6_3(x >> 2))
}

/// Spreads 8 bits into every-fourth bit position of a 32-bit value.
#[inline(always)]
pub const fn spread8_4(x: u8) -> u32 {
    let mut x = x as u32;
    x = (x | (x << 12)) & 0x000F_000F;
    x = (x | (x << 6)) & 0x0303_0303;
    x = (x | (x << 3)) & 0x1111_1111;
    x
}

/// Interleaves the bits of four 8-bit values into a 32-bit Morton code.
#[inline(always)]
pub const fn interleave8_4(x: u8, y: u8, z: u8, w: u8) -> u32 {
    spread8_4(x) | (spread8_4(y) << 1) | (spread8_4(z) << 2) | (spread8_4(w) << 3)
}

/// Extracts every-fourth bit from a 32-bit value back into 8 bits.
#[inline(always)]
pub const fn compact8_4(x: u32) -> u8 {
    let mut x = x;
    x = x & 0x1111_1111;
    x = (x | (x >> 3)) & 0x0303_0303;
    x = (x | (x >> 6)) & 0x000F_000F;
    x = x | (x >> 12);
    x as u8
}

/// Deinterleaves a 32-bit Morton code into four 8-bit values.
#[inline(always)]
pub const fn deinterleave8_4(x: u32) -> (u8, u8, u8, u8) {
    (
        compact8_4(x),
        compact8_4(x >> 1),
        compact8_4(x >> 2),
        compact8_4(x >> 3),
    )
}
