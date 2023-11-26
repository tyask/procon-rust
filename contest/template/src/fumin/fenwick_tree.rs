#![allow(dead_code)]
use std::{*, iter::FromIterator, ops::{AddAssign, Sub, SubAssign, RangeBounds}};
use ::num::Zero;
use crate::common::*;
use super::range_bounds_ex::RangeBoundsEx;

// CAP(fumin::range_bounds_ex)

pub trait E:
      Clone
    + Copy
    + Zero
    + AddAssign<Self>
    + Sub<Output=Self>
    + SubAssign<Self>
    + PartialOrd
{}

impl<T> E for T where 
    T: Clone
    + Copy
    + Zero
    + AddAssign<Self>
    + Sub<Output=Self>
    + SubAssign<Self>
    + PartialOrd
 {}

// 0-indexed
pub struct FenwickTree<T> {
    dat: Vec<T>
}

impl<T: E> FenwickTree<T> {
    pub fn new(n: us) -> Self { Self { dat: vec![T::zero(); n+1] } }

    // O(logN)
    pub fn add(&mut self, mut p: us, v: T) {
        assert!(p < self.len());
        p += 1;
        while p <= self.len() { self.dat[p-1] += v; p += Self::lsb(p); }
    }

    pub fn set(&mut self, p: us, v: T) {
        self.add(p, v-self.get(p))
    }

    // sum of [0, p) O(logN)
    pub fn sum0(&self, mut p: us) -> T {
        assert!(p < self.len());
        let mut r = T::zero();
        while p > 0 { r += self.dat[p-1]; p -= Self::lsb(p); }
        r
    }

    pub fn sum(&self, r: impl RangeBounds<us>) -> T {
        let (l, r) = r.clamp(0, self.len() - 1);
        self.sum0(r) - self.sum0(l)
    }

    pub fn get(&self, p: us) -> T {
        self.sum(p..p+1)
    }

    pub fn lower_bound(&self, mut v: T) -> us {
        if v <= T::zero() { return 0; }
        let (mut x, mut k) = (0, self.len().next_power_of_two()>>1);
        while k > 0 {
            if x+k <= self.len() && self.dat[x+k-1] < v { v -= self.dat[x+k-1]; x += k; }
            k /= 2;
        }
        x
    }

    // least significant bit (引数を割り切る最大の2冪)
    fn lsb(n: us) -> us { let n = n as i64; (n & -n) as us }
    fn len(&self) -> us { self.dat.len() }
}

impl<T: E> FromIterator<T> for FenwickTree<T> {
    fn from_iter<It: IntoIterator<Item=T>>(iter: It) -> Self {
        let v = iter.into_iter().cv();
        let mut t = Self::new(v.len());
        for (i, &x) in v.iter().enumerate() { t.add(i, x); }
        t
    }
}

impl<T: E> From<Vec<T>> for FenwickTree<T> {
    fn from(value: Vec<T>) -> Self { Self::from_iter(value.into_iter()) }
}

// 転倒数
// 各iについてv[i]<v.len()である必要がある. そうでない場合は事前に座圧しておく.
pub fn count_inversions(v: &Vec<i64>) -> us {
    let mut c = 0;
    let mut t = FenwickTree::<i64>::new(v.len()); 
    for &x in v {
        let x = x.us();
        c += t.sum(x+1..).us();
        t.add(x, 1);
    }
    c
}

