#![allow(dead_code)]

use std::ops::{Add, Sub};

pub trait Filterable: Add<Output = Self> + Sub<Output = Self> + Copy + Sized {
    type Distance: Ord + Copy + Sized;
    fn distance(lhs: &Self, rhs: &Self) -> Self::Distance;
}

// Paeth filter is based on an algorithm by Alan W. Paeth
pub fn filter_paeth<T>(a: T, b: T, c: T) -> T
where
    T: Filterable,
{
    let p = a + b - c;

    let pa = T::distance(&p, &a);
    let pb = T::distance(&p, &b);
    let pc = T::distance(&p, &c);

    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}
