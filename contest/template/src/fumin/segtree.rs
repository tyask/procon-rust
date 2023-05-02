#![allow(dead_code)]
use std::{ops::{*, self}, convert::Infallible, marker::PhantomData, cmp::min, iter::FromIterator};
use num::Zero;
use crate::common::*;

pub trait Monoid {
    type T: Copy;
    fn op(a: Self::T, b: Self::T) -> Self::T;
    fn e() -> Self::T;
}

#[derive(Clone)]
pub struct SegTree<M: Monoid> {
    n: us, // 実データのサイズ
    log: us, 
    size: us,
    dat: Vec<M::T>,
}

impl<M: Monoid> SegTree<M> {

    pub fn set(&mut self, mut p: us, v: M::T) {
        self.check_index(p);
        p += self.size;
        self.dat[p] = v;
        for i in 1..=self.log {
            self.update(p >> i);
        }
    }

    fn update(&mut self, k: us) {
        self.dat[k] = M::op(self.dat[2*k], self.dat[2*k+1]);
    }

    fn check_index(&self, p: us) {
        assert!(p < self.n);
    }

    pub fn get(&self, p: us) -> M::T {
        self.check_index(p);
        self.dat[p + self.size]
    }

    pub fn prod(&self, rng: impl RangeBounds<us>) -> M::T {
        let rng = self.to_range(&rng);
        let (mut l, mut r) = (rng.start, rng.end);
        assert!(l <= r && r <= self.n);
        let mut sml = M::e();
        let mut smr = M::e();
        l += self.size;
        r += self.size;

        while l < r {
            if l & 1 == 1 {
                sml = M::op(sml, self.dat[l]);
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                smr = M::op(self.dat[r], smr);
            }
            l >>= 1;
            r >>= 1;
        }

        M::op(sml, smr)

    }

    fn prod_sub(&self, rng: &impl RangeBounds<us>, k: us, l: us, r: us) -> M::T {
        let rng = self.to_range(rng);
        let (a, b) = (rng.start, rng.end);

        if r <= a || b <= l { return M::e(); }
        if a <= l && r <= b { return self.dat[k]; }

        let vl = self.prod_sub(&rng, k*2+1, l, (l+r)/2);
        let vr = self.prod_sub(&rng, k*2+2, (l+r)/2, r);
        M::op(vl, vr)
    }

    fn to_range(&self, r: &impl RangeBounds<us>) -> Range<us> {
        let s = match r.start_bound() {
            Bound::Included(&x) => x,
            Bound::Excluded(&x) => x + 1,
            Bound::Unbounded    => 0,
        };
        let e = match r.end_bound() {
            Bound::Included(&x) => x + 1,
            Bound::Excluded(&x) => x,
            Bound::Unbounded    => self.n,
        };
        s..e
    }

}

impl <M: Monoid> From<&Vec<M::T>> for SegTree<M> {
    fn from(v: &Vec<M::T>) -> Self {
        let n = v.len();
        let log = ceilpow2(n);
        let size = 1 << log;
        let mut dat = vec![M::e(); 2 * size];
        dat[size..][..n].clone_from_slice(v);
        let mut s = SegTree { n, log, size, dat };
        for i in (0..size-1).rev() { s.update(i); }
        s
    }
}

impl<M: Monoid> SegTree<M> {
    pub fn new(n: us) -> Self { vec![M::e(); n].into() }
}
impl <M: Monoid> From<Vec<M::T>> for SegTree<M> {
    fn from(v: Vec<M::T>) -> Self { (&v).into() }
}
impl <M: Monoid> FromIterator<M::T> for SegTree<M> {
    fn from_iter<T: IntoIterator<Item = M::T>>(iter: T) -> Self {
        iter.into_iter().cv().into()
    }
}

impl<M: Monoid> SegTree<M> where M::T: Copy+Add<Output=M::T> {
    pub fn add(&mut self, i: us, v: M::T) { self.set(i, self.get(i) + v); }
}
impl<M: Monoid> SegTree<M> where M::T: Copy+Sub<Output=M::T> {
    pub fn sub(&mut self, i: us, v: M::T) { self.set(i, self.get(i) - v); }
}

#[derive(Clone)]
pub struct Min<T>(Infallible, PhantomData<fn() -> T>);
impl<T: Copy + Ord + Inf> Monoid for Min<T> {
    type T = T;
    fn op(a: Self::T, b: Self::T) -> Self::T { min(a, b) }
    fn e() -> Self::T { T::INF }
}

#[derive(Clone)]
pub struct Max<T>(Infallible, PhantomData<fn() -> T>);
impl<T: Copy + Ord + Inf> Monoid for Max<T> {
    type T = T;
    fn op(a: Self::T, b: Self::T) -> Self::T { std::cmp::max(a, b) }
    fn e() -> Self::T { T::MINF }
}

#[derive(Clone)]
pub struct Additive<T>(Infallible, PhantomData<fn() -> T>);
impl<T: Copy + Zero> Monoid for Additive<T> {
    type T = T;
    fn op(a: Self::T, b: Self::T) -> Self::T { a + b }
    fn e() -> Self::T { T::zero() }
}

#[derive(Clone)]
pub struct BitOr<T>(Infallible, PhantomData<fn() -> T>);
impl<T: Copy + Ord + Zero + ops::BitOr<Output=T>> Monoid for BitOr<T> {
    type T = T;
    fn op(a: Self::T, b: Self::T) -> Self::T { a | b }
    fn e() -> Self::T { T::zero() }
}

fn ceilpow2(n: us) -> us { std::mem::size_of::<us>() * 8 - n.saturating_sub(1).leading_zeros().us() }

// CAP(IGNORE_BELOW)

#[cfg(test)]
mod tests {
    use std::ops::RangeBounds;
    use crate::common::*;
    use core::ops::Bound::{Excluded, Included};
    use super::{SegTree, Max};

    #[test]
    fn test_max_segtree() {
        let base: Vec<i64> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let n = base.len();
        let segtree: SegTree<Max<_>> = base.clone().into();
        check_segtree(&base, &segtree);

        let mut segtree = SegTree::<Max<_>>::new(n);
        let mut internal = vec![i64::MINF; n];
        for i in 0..n {
            segtree.set(i, base[i]);
            internal[i] = base[i];
            check_segtree(&internal, &segtree);
        }

        segtree.set(6, 5);
        internal[6] = 5;
        check_segtree(&internal, &segtree);

        segtree.set(6, 0);
        internal[6] = 0;
        check_segtree(&internal, &segtree);
    }

    fn check_segtree(base: &[i64], segtree: &SegTree<Max<i64>>) {
        let n = base.len();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            assert_eq!(segtree.get(i), base[i], "i={}", i);
        }

        check(base, segtree, ..);
        for i in 0..=n {
            check(base, segtree, ..i);
            check(base, segtree, i..);
            if i < n {
                check(base, segtree, ..=i);
            }
            for j in i..=n {
                check(base, segtree, i..j);
                if j < n {
                    check(base, segtree, i..=j);
                    check(base, segtree, (Excluded(i), Included(j)));
                }
            }
        }
    }

    fn check(base: &[i64], segtree: &SegTree<Max<i64>>, range: impl RangeBounds<usize>) {
        let expected = base
            .iter()
            .enumerate()
            .filter_map(|(i, a)| Some(a).filter(|_| range.contains(&i)))
            .max()
            .copied()
            .unwrap_or(i64::MINF);
        assert_eq!(segtree.prod(range), expected);
    }
}