//! All the math functions are implemented here.

use std::{
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    // This lerp is monotonic and produces exactly a for t = 0 and b for t = 1.

    if t < 0.5 {
        (b - a).mul_add(t, a)
    } else {
        (a - b).mul_add(1.0 - t, b)
    }
}

/// A 2D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec2([f32; 2]);

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
    pub const ZERO: Vec2 = Vec2([0.0, 0.0]);

    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Vec2([x, y])
    }

    #[inline(always)]
    pub const fn splat(value: f32) -> Self {
        Vec2([value, value])
    }

    #[inline(always)]
    pub const fn dot(self, rhs: Vec2) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y()
    }

    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.dot(*self).sqrt()
    }

    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

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

    #[inline(always)]
    pub const fn with_z(&self, z: f32) -> Vec3 {
        Vec3([self.x(), self.y(), z])
    }

    #[inline(always)]
    pub const fn with_zw(&self, z: f32, w: f32) -> Vec4 {
        Vec4([self.x(), self.y(), z, w])
    }

    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Vec2([lerp(a.x(), b.x(), t), lerp(a.y(), b.y(), t)])
    }
}

/// A 3D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec3([f32; 3]);

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
    pub const ZERO: Vec3 = Vec3([0.0, 0.0, 0.0]);

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }

    pub const fn splat(value: f32) -> Self {
        Vec3([value, value, value])
    }

    #[inline(always)]
    pub const fn dot(self, rhs: Vec3) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z()
    }

    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.dot(*self).sqrt()
    }

    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

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

    #[inline(always)]
    pub const fn with_w(&self, w: f32) -> Vec4 {
        Vec4([self.x(), self.y(), self.z(), w])
    }

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
    pub const ZERO: Vec4 = Vec4([0.0, 0.0, 0.0, 0.0]);

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4([x, y, z, w])
    }

    #[inline(always)]
    pub const fn splat(value: f32) -> Self {
        Vec4([value, value, value, value])
    }

    #[inline(always)]
    pub const fn dot(self, rhs: Vec4) -> f32 {
        self.x() * rhs.x() + self.y() * rhs.y() + self.z() * rhs.z() + self.w() * rhs.w()
    }

    #[inline(always)]
    pub fn length(&self) -> f32 {
        self.dot(*self).sqrt()
    }

    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub const fn z(&self) -> f32 {
        self.0[2]
    }

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
    pub min: Vec3,
    pub max: Vec3,
}

impl Region3 {
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

    pub fn min(&self) -> Vec3 {
        self.min
    }

    pub fn max(&self) -> Vec3 {
        self.max
    }

    pub fn is_empty(&self) -> bool {
        self.min.x() > self.max.x() || self.min.y() > self.max.y() || self.min.z() > self.max.z()
    }

    pub fn is_singular(&self) -> bool {
        self.min == self.max
    }

    /// Returns 4 diagonals of the region.
    pub fn diagonals(&self) -> [(Vec3, Vec3); 4] {
        let diagonals = [
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
        ];

        diagonals
    }

    pub fn is_real(&self) -> bool {
        self.min.x() <= self.max.x() && self.min.y() <= self.max.y() && self.min.z() <= self.max.z()
    }

    pub fn volume(&self) -> f32 {
        let diff = self.max - self.min;
        diff.x().min(0.0) * diff.y().min(0.0) * diff.z().min(0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Luma8U(u8);

impl Luma8U {
    pub const WHITE: Luma8U = Luma8U(255);
    pub const BLACK: Luma8U = Luma8U(0);

    #[inline(always)]
    pub const fn new(l: u8) -> Self {
        Luma8U(l)
    }

    #[inline(always)]
    pub const fn l(&self) -> u8 {
        self.0
    }

    #[inline(always)]
    pub const fn bits(&self) -> u8 {
        self.0
    }

    #[inline(always)]
    pub fn wrapping_add(self, other: Self) -> Self {
        Luma8U(self.0.wrapping_add(other.0))
    }

    #[inline(always)]
    pub fn wrapping_sub(self, other: Self) -> Self {
        Luma8U(self.0.wrapping_sub(other.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Luma32F(f32);

impl Luma32F {
    pub const WHITE: Luma32F = Luma32F(1.0);
    pub const BLACK: Luma32F = Luma32F(0.0);

    #[inline(always)]
    pub const fn new(l: f32) -> Self {
        Luma32F(l)
    }

    #[inline(always)]
    pub const fn l(&self) -> f32 {
        self.0
    }

    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Luma32F(lerp(a.l(), b.l(), t))
    }

    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> f32 {
        a.l() - b.l()
    }

    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff * diff
    }

    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        a.l() - b.l()
    }

    #[inline(always)]
    pub const fn offset(self, offset: f32) -> Self {
        Luma32F(self.l() + offset)
    }
}

/// An RGB color represented as 3 floats.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Rgb32F([f32; 3]);

impl Rgb32F {
    pub const WHITE: Rgb32F = Rgb32F([1.0, 1.0, 1.0]);
    pub const BLACK: Rgb32F = Rgb32F([0.0, 0.0, 0.0]);

    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Rgb32F([r, g, b])
    }

    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub const fn g(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub const fn b(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub const fn with_alpha(self, a: f32) -> Rgba32F {
        Rgba32F([self.r(), self.g(), self.b(), a])
    }

    #[inline(always)]
    pub const fn into_opaque(self) -> Rgba32F {
        Rgba32F([self.r(), self.g(), self.b(), 1.0])
    }

    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Rgb32F([
            lerp(a.r(), b.r(), t),
            lerp(a.g(), b.g(), t),
            lerp(a.b(), b.b(), t),
        ])
    }

    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> Vec3 {
        Vec3([a.r() - b.r(), a.g() - b.g(), a.b() - b.b()])
    }

    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff.dot(diff)
    }

    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }

    #[inline(always)]
    pub const fn offset(self, offset: Vec3) -> Self {
        Rgb32F([
            self.r() + offset.x(),
            self.g() + offset.y(),
            self.b() + offset.z(),
        ])
    }
}

/// An RGB color represented as 3 floats.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Rgba32F([f32; 4]);

impl Rgba32F {
    pub const WHITE: Rgba32F = Rgba32F([1.0, 1.0, 1.0, 1.0]);
    pub const BLACK: Rgba32F = Rgba32F([0.0, 0.0, 0.0, 1.0]);
    pub const TRANSPARENT: Rgba32F = Rgba32F([0.0, 0.0, 0.0, 0.0]);

    #[inline(always)]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Rgba32F([r, g, b, a])
    }

    #[inline(always)]
    pub const fn r(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub const fn g(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub const fn b(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub const fn a(&self) -> f32 {
        self.0[3]
    }

    #[inline(always)]
    pub const fn rgb(&self) -> Rgb32F {
        Rgb32F([self.r(), self.g(), self.b()])
    }

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

/// An RGB color with 8 bit unsigned normalized integers per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rgb8U([u8; 3]);

/// An RGB color with 5,6 and 5 bits unsigned normalized integers per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Rgb565(u16);

impl Rgb565 {
    pub const WHITE: Rgb565 = Rgb565(0b11111_111111_11111);
    pub const BLACK: Rgb565 = Rgb565(0);

    pub fn new(r: u8, g: u8, b: u8) -> Self {
        assert!(r <= 31, "Red channel must be in range 0..=31");
        assert!(g <= 63, "Green channel must be in range 0..=63");
        assert!(b <= 31, "Blue channel must be in range 0..=31");

        let r = r as u16;
        let g = g as u16;
        let b = b as u16;
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

    #[inline(always)]
    pub fn r(&self) -> u8 {
        (self.0 >> 11) as u8
    }

    #[inline(always)]
    pub fn g(&self) -> u8 {
        ((self.0 >> 5) & 0b111111) as u8
    }

    #[inline(always)]
    pub fn b(&self) -> u8 {
        (self.0 & 0b11111) as u8
    }

    #[inline(always)]
    pub fn set_r(&mut self, r: u8) {
        assert!(r <= 31, "Red channel must be in range 0..=31");
        self.0 = (self.0 & 0b00000_111111_11111) | ((r as u16) << 11);
    }

    #[inline(always)]
    pub fn set_g(&mut self, g: u8) {
        assert!(g <= 63, "Green channel must be in range 0..=63");
        self.0 = (self.0 & 0b11111_000000_11111) | ((g as u16) << 5);
    }

    #[inline(always)]
    pub fn set_b(&mut self, b: u8) {
        assert!(b <= 31, "Blue channel must be in range 0..=31");
        self.0 = (self.0 & 0b11111_111111_00000) | (b as u16);
    }

    #[inline(always)]
    pub const fn into_f32(self) -> Rgb32F {
        let r = ((self.0 >> 11) & 0b11111) as f32 / 31.0;
        let g = ((self.0 >> 5) & 0b111111) as f32 / 63.0;
        let b = (self.0 & 0b11111) as f32 / 31.0;
        Rgb32F([r, g, b])
    }

    #[inline(always)]
    pub fn from_f32(rgb: Rgb32F) -> Self {
        let [r, g, b] = rgb.0;
        let r = (r * 31.0).clamp(0.0, 31.0) as u16;
        let g = (g * 63.0).clamp(0.0, 63.0) as u16;
        let b = (b * 31.0).clamp(0.0, 31.0) as u16;
        Rgb565((r << 11) | (g << 5) | b)
    }

    pub fn wrapping_add(a: Self, b: Self) -> Self {
        let r = a.r().wrapping_add(b.r()) & 31;
        let g = a.g().wrapping_add(b.g()) & 63;
        let b = a.b().wrapping_add(b.b()) & 31;
        Rgb565::new(r, g, b)
    }

    pub fn wrapping_sub(a: Self, b: Self) -> Self {
        let r = a.r().wrapping_sub(b.r()) & 31;
        let g = a.g().wrapping_sub(b.g()) & 63;
        let b = a.b().wrapping_sub(b.b()) & 31;
        Rgb565::new(r, g, b)
    }
}

impl From<Rgb565> for Rgb32F {
    #[inline(always)]
    fn from(rgb: Rgb565) -> Self {
        rgb.into_f32()
    }
}

impl From<Rgb32F> for Rgb565 {
    #[inline(always)]
    fn from(rgb: Rgb32F) -> Self {
        Rgb565::from_f32(rgb)
    }
}

/// An YIQ color represented as 3 floats.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Yiq32F([f32; 3]);

impl Yiq32F {
    pub const WHITE: Yiq32F = Yiq32F([1.0, 0.0, 0.0]);
    pub const BLACK: Yiq32F = Yiq32F([0.0, 0.0, 0.0]);

    #[inline(always)]
    pub const fn new(y: f32, i: f32, q: f32) -> Self {
        Yiq32F([y, i, q])
    }

    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub const fn i(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub const fn q(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub const fn from_rgb(rgb: Rgb32F) -> Self {
        let [r, g, b] = rgb.0;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let i = 0.5959 * r - 0.2746 * g - 0.3213 * b;
        let q = 0.2115 * r - 0.5227 * g + 0.3112 * b;
        Yiq32F([y, i, q])
    }

    #[inline(always)]
    pub const fn into_rgb(self) -> Rgb32F {
        let [y, i, q] = self.0;
        let r = y + 0.956 * i + 0.619 * q;
        let g = y - 0.272 * i - 0.647 * q;
        let b = y - 1.106 * i + 1.703 * q;
        Rgb32F([r, g, b])
    }

    #[inline(always)]
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Yiq32F([
            lerp(a.y(), b.y(), t),
            lerp(a.i(), b.i(), t),
            lerp(a.q(), b.q(), t),
        ])
    }

    #[inline(always)]
    pub fn perceptual_distance(a: Self, b: Self) -> f32 {
        let [y1, i1, q1] = a.0;
        let [y2, i2, q2] = b.0;

        let luminance_diff = (y1 - y2) * (y1 - y2);
        let chrominance_diff = 0.25 * ((i1 - i2) * (i1 - i2) + (q1 - q2) * (q1 - q2));

        (luminance_diff + chrominance_diff).sqrt()
    }

    #[inline(always)]
    pub const fn diff(a: Self, b: Self) -> Vec3 {
        Vec3([a.y() - b.y(), a.i() - b.i(), a.q() - b.q()])
    }

    #[inline(always)]
    pub const fn distance_squared(a: Self, b: Self) -> f32 {
        let diff = Self::diff(a, b);
        diff.dot(diff)
    }

    #[inline(always)]
    pub fn distance(a: Self, b: Self) -> f32 {
        Self::distance_squared(a, b).sqrt()
    }

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

pub(crate) fn predict_color_u8(left: u8, top: u8, top_left: u8) -> u8 {
    // let target = left.wrapping_add(top).wrapping_sub(top_left);

    // let left_dist = if left > target {
    //     left - target
    // } else {
    //     target - left
    // };

    // let top_dist = if top > target {
    //     top - target
    // } else {
    //     target - top
    // };

    // let top_left = if top_left > target {
    //     top_left - target
    // } else {
    //     target - top_left
    // };

    // if left_dist < top_dist {
    //     if left_dist < top_left {
    //         return left;
    //     } else {
    //         return top_left;
    //     }
    // } else {
    //     if top_dist < top_left {
    //         return top;
    //     } else {
    //         return top_left;
    //     }
    // }

    // let v_diff = if top_left > left {
    //     top_left - left
    // } else {
    //     left - top_left
    // };

    // let h_diff = if top_left > top {
    //     top_left - top
    // } else {
    //     top - top_left
    // };

    // if v_diff > 10 || h_diff > 10 {
    if top_left > left && top_left > top {
        if top >= left {
            top
        } else {
            left
        }
    } else if top_left < left && top_left < top {
        if top >= left {
            left
        } else {
            top
        }
    } else {
        left.wrapping_add(top).wrapping_sub(top_left)
    }
    // } else {
    //     left.wrapping_add(top).wrapping_sub(top_left)
    // }
}

pub(crate) trait PredictableColor: Copy {
    fn predict_color(left: Self, top: Self, top_left: Self) -> Self;
}

impl PredictableColor for Rgb565 {
    fn predict_color(left: Self, top: Self, top_left: Self) -> Self {
        Rgb565::new(
            predict_color_u8(left.r(), top.r(), top_left.r()) & 31,
            predict_color_u8(left.g(), top.g(), top_left.g()) & 63,
            predict_color_u8(left.b(), top.b(), top_left.b()) & 31,
        )
    }
}
