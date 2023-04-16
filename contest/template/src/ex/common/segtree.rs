#![allow(dead_code)]
use std::ops::*;
use crate::common::*;

pub struct SegTree<T> {
    n: us,
    dat: Vec<T>,
    fx: Box<dyn Fn(T,T)->T>,
    e: T,
}

impl<T: Copy> SegTree<T> {
    pub fn new(n: us, fx: Box<dyn Fn(T,T)->T>, e: T) -> Self {
        Self{n: ceilpow2(n), dat: vec![e; n*4], fx, e}
    }

    pub fn update(&mut self, mut i: us, v: T) {
        i += self.n - 1;
        self.dat[i] = v;
        while i > 0 {
            i = par(i);
            self.dat[i] = self.fx.as_ref()(self.dat[lch(i)], self.dat[rch(i)]);
        }
    }

    pub fn query(&self, i: us) -> T { self.query_rng(i..=i) }
    pub fn query_rng(&self, rng: impl RangeBounds<us>) -> T { self.query_sub(&rng, 0, 0, self.n) }

    fn query_sub(&self, rng: &impl RangeBounds<us>, k: us, l: us, r: us) -> T {
        let rng = self.to_range(rng);
        let (a, b) = (rng.start, rng.end);

        if r <= a || b <= l { return self.e; }
        if a <= l && r <= b { return self.dat[k]; }

        let vl = self.query_sub(&rng, k*2+1, l, (l+r)/2);
        let vr = self.query_sub(&rng, k*2+2, (l+r)/2, r);
        self.fx.as_ref()(vl, vr)
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

impl<T: Copy+Add<Output=T>> SegTree<T> {
    pub fn add(&mut self, i: us, v: T) { self.update(i, self.query(i) + v); }
}
impl<T: Copy+Sub<Output=T>> SegTree<T> {
    pub fn sub(&mut self, i: us, v: T) { self.update(i, self.query(i) - v); }
}

impl<T: Copy+Ord> SegTree<T> {
    pub fn rng_min_query(n: us, max: T) -> Self {
        Self::new(n, Box::new(|a,b|std::cmp::min(a,b)), max)
    }
    pub fn rng_max_query(n: us, min: T) -> Self {
        Self::new(n, Box::new(|a,b|std::cmp::max(a,b)), min)
    }
}
impl<T: Copy+Add<Output=T>+Default> SegTree<T> {
    pub fn rng_sum_query(n: us) -> Self {
        Self::new(n, Box::new(|a,b|a+b), T::default())
    }
}

fn ceilpow2(cap: us) -> us { let mut x = 1; while cap > x { x *= 2; } x }
fn par(i: us) -> us { (i - 1) / 2 }
fn lch(i: us) -> us { i * 2 + 1 }
fn rch(i: us) -> us { i * 2 + 2 }
