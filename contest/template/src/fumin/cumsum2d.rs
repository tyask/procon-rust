#![allow(dead_code)]
use std::ops::{RangeBounds, Add, Sub};
use num::Zero;
use crate::common::*;
use super::range_bounds_ex::RangeBoundsEx;

// CAP(fumin::range_bounds_ex)

pub struct CumSum2d<N> { pub s: Vec<Vec<N>> }

impl<N: Zero+Copy+Add<Output=N>+Sub<Output=N>> CumSum2d<N> {
    pub fn new(v: &Vec<Vec<N>>) -> Self {
        let mut s = vec![vec![N::zero(); v[0].len()+1]; v.len()+1];
        for i in 0..v.len() { for j in 0..v[i].len() {
            s[i+1][j+1] = s[i][j+1] + s[i+1][j] - s[i][j] + v[i][j];
        }}
        Self { s }
    }
    pub fn sum(&self, ir: impl RangeBounds<us>, jr: impl RangeBounds<us>) -> N {
        let s = &self.s;
        let (i1, i2) = ir.clamp(0, s.len());
        let (j1, j2) = jr.clamp(0, s[0].len());
        s[i2][j2] - s[i1][j2] - s[i2][j1] + s[i1][j1]
    }

    pub fn sum0(&self, i: us, j: us) -> N { self.sum(..i, ..j) }
}
