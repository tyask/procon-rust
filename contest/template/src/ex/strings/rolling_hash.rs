#![allow(dead_code)]
use std::ops::{RangeBounds, Bound};
use crate::common::IntoT;

const MASK30      : u64 = (1 << 30) - 1;
const MASK31      : u64 = (1 << 31) - 1;
const MASK61      : u64 = (1 << 61) - 1;
const MOD         : u64 = (1 << 61) - 1;
const POSITIVIZER : u64 = MOD * 4;
const BASE        : u64 = 1_000_000_009;


pub struct RollingHash {
    hash: Vec<u64>,
    pos: Vec<u64>,
}

impl RollingHash {

    pub fn new<T: Copy+IntoT<u64>>(s: &[T]) -> Self {
        let mut hash = vec![0];
        let mut pos = vec![1];
        for (i, &x) in s.iter().enumerate() {
            hash.push(mods(mul(hash[i], BASE) + x.into_t()));
            pos.push(mods(mul(pos[i], BASE)));
        }
        Self { hash, pos }
    }

    pub fn hash<R: RangeBounds<usize>>(&self, rng: R) -> u64 {
        let (l, r) = self.resolve_rng(&rng);
        mods(self.hash[r] + POSITIVIZER - mul(self.hash[l], self.pos[r-l]))
    }

    pub fn conn<RS: RangeBounds<usize>, RE: RangeBounds<usize>>(&self, l: RS, r: RE) -> u64 {
        let (_, le) = self.resolve_rng(&l);
        let (rs, re) = self.resolve_rng(&r);
        assert!(le <= rs);
        mods(mul(self.hash(l), self.pos[re-rs]) + self.hash(r))
    }

    fn resolve_rng<R: RangeBounds<usize>>(&self, rng: &R) -> (usize, usize) {
        let l = match rng.start_bound() {
            Bound::Included(&x) => x,
            Bound::Excluded(&x) => x + 1,
            Bound::Unbounded    => 0,
        };
        let r = match rng.end_bound() {
            Bound::Included(&x) => x + 1,
            Bound::Excluded(&x) => x,
            Bound::Unbounded    => self.hash.len() - 1,
        };
        (l, r)
    }

}

// 64bit内で掛け算
fn mul(a: u64, b: u64) -> u64 {
    let (au, ad, bu, bd) = (a>>31, a&MASK31, b>>31, b&MASK31);
    let m = ad * bu + au * bd;
    let (mu, md) = (m >> 30, m&MASK30);
    au*bu*2 + mu + (md<<31) + ad*bd
}

fn mods(x: u64) -> u64 {
    let (xu, xd) = (x>>61, x&MASK61);
    let mut a = xu + xd;
    if a >= MOD { a -= MOD; }
    a
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod test {
    use super::RollingHash;

    #[test]
    fn test() {
        let s = "abcdefg".chars().collect::<Vec<_>>();
        let h = RollingHash::new(&s);
        assert_eq!(h.hash(0..s.len()), 1522289676931369548);
    }
}