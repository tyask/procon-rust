#![allow(dead_code)]
use std::ops::RemAssign;
use alga::general::{ClosedMul, ClosedAdd};
use itertools::iproduct;
use nalgebra::{*, allocator::Allocator};
use num::{Zero, One};
use crate::common::*;

trait MatrixTrait<N> {
    // self^k mod m
    fn powmod(&self, k: impl IntoT<us>, m: N) -> Self;

    // self*a mod m
    fn mulmod(&self, a: &Self, m: N) -> Self;
}

impl<N, D> MatrixTrait<N> for MatrixN<N, D>
where
    N: Scalar + Zero + One + ClosedMul + ClosedAdd + RemAssign + Copy,
    D: DimName,
    DefaultAllocator: Allocator<N, D, D>,
{
    fn powmod(&self, k: impl IntoT<us>, m: N) -> Self {
        let (mut a, mut k, mut r) = (Self::clone(self), k.into_t(), Self::one());
        while k > 0 {
            if k & 1 == 1 { r = r.mulmod(&a, m); }
            a = a.mulmod(&a, m);
            k >>= 1;
        }
        r
    }

    fn mulmod(&self, a: &Self, m: N) -> Self {
        let (n, mut r) = (self.nrows(), Self::zero());
        for (i, j, k) in iproduct!(0..n, 0..n, 0..n) {
            r[(i,j)] += self[(i,k)] * a[(k,j)];
            r[(i,j)] %= m;
        }
        r
    }
}