#![allow(dead_code)]
use crate::common::*;

pub struct CumSum<N> { pub s: Vec<N> }
impl<N: SimplePrimInt + FromT<us>> CumSum<N> {
    pub fn new(v: &Vec<N>) -> Self {
        let mut s = vec![N::from_t(0); v.len() + 1];
        for i in 0..v.len() { s[i+1] = s[i] + v[i]; }
        Self { s }
    }
    pub fn sum(&self, l: us, r: us) -> N { self.s[r] - self.s[l] }
}

pub struct CumSum2d<N> { pub s: Vec<Vec<N>> }
impl<N: SimplePrimInt+FromT<us>> CumSum2d<N> {
    pub fn new(v: &Vec<Vec<N>>) -> Self {
        let mut s = vec![vec![N::from_t(0); v[0].len()+1]; v.len()+1];
        for i in 0..v.len() { for j in 0..v[i].len() {
            s[i+1][j+1] = s[i][j+1] + s[i+1][j] - s[i][j] + v[i][j];
        }}
        Self { s }
    }
    pub fn sum(&self, i1: us, i2: us, j1: us, j2: us) -> N {
        let s = &self.s;
        s[i2][j2] - s[i1][j2] - s[i2][j1] + s[i1][j1]
    }
    pub fn sum0(&self, i: us, j: us) -> N { self.sum(0, i, 0, j) }
}
