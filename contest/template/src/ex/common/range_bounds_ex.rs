#![allow(dead_code)]
use std::{ops::{RangeBounds, Add}, cmp};
use num::One;

pub trait RangeBoundsEx<T: Copy+Ord+Add<Output=T>+One>: RangeBounds<T> {
    // このrangeを[low,high]内に収める
    fn clamp(&self, low: T, high: T) -> (T, T) {
        let s = match self.start_bound() {
            std::ops::Bound::Included(&v) => v,
            std::ops::Bound::Excluded(&v) => v + T::one(),
            std::ops::Bound::Unbounded    => low,
        }; 
        let e = match self.end_bound() {
            std::ops::Bound::Included(&v) => v + T::one(),
            std::ops::Bound::Excluded(&v) => v,
            std::ops::Bound::Unbounded    => high,
        }; 
        (cmp::max(s, low), cmp::min(e, high))
    }
}

impl<T:Copy+Ord+Add<Output=T>+One, R: RangeBounds<T>> RangeBoundsEx<T> for R { }
