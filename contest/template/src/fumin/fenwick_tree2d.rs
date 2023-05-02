
#![allow(dead_code)]
use std::*;
use crate::common::*;

struct FenwickTree2d {
    n: us,
    m: us,
    dat: Vec<Vec<is>>
}

impl FenwickTree2d {
    pub fn new(n: us, m: us) -> Self { Self{ n: n, m: m, dat: vec![vec![0; m]; n] } }

    // O(logN)
    pub fn add(&mut self, i: us, j: us, v: is) {
        let (n, m) = (self.n, self.m);
        let (mut i, mut j) = (i+1, j+1);
        while i <= n {
            while j <= m { self.dat[i-1][j-1] += v; j += lsb(j); }
            i += lsb(i);
        }
    }

    // sum of [0, p) O(logN)
    pub fn sum(&self, mut i: us, mut j: us) -> is {
        let mut r = 0;
        while i > 0 {
            while j > 0 { r += self.dat[i-1][j-1]; j -= lsb(j); }
            i -= lsb(i);
        }
        r
    }

    pub fn sum2(&self, i1: us, j1: us, i2: us, j2: us) -> is {
        self.sum(i2, j2) - self.sum(i1, j1)
    }
}

// least significant bit (引数を割り切る最大の2冪)
fn lsb(n: us) -> us{ let n = n as is; (n & -n) as us}
