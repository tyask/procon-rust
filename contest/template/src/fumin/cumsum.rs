#![allow(dead_code)]
use std::ops::{RangeBounds, Sub};
use num::Zero;
use crate::common::*;
use super::range_bounds_ex::RangeBoundsEx;

// CAP(fumin::range_bounds_ex)
pub struct CumSum<N> { pub s: Vec<N> }
impl<N: Clone+Copy+Zero+Sub<Output=N>> CumSum<N> {
    pub fn new(v: &[N]) -> Self {
        let mut s = vec![N::zero(); v.len() + 1];
        for i in 0..v.len() { s[i+1] = s[i] + v[i]; }
        Self { s }
    }
    pub fn sum(&self, r: impl RangeBounds<us>) -> N {
        let (l, r) = r.clamp(0, self.s.len() - 1);
        self.s[r] - self.s[l]
    }
}
