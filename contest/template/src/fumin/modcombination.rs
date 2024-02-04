#![allow(dead_code)]
use std::ops::Sub;
use num::{Zero, One};
use crate::common::*;
use super::modint::Modint;

// CAP(fumin::modint)

pub struct ModCombination<M> {
    facts: Vec<M>,
    ifacts: Vec<M>
}

impl<const M: i64> ModCombination<Modint<M>> {
    pub fn new(n: us) -> Self {
        assert!(n.i64() < M);
        let mut facts  = vec![Self::zero(); n+1];
        let mut ifacts = vec![Self::zero(); n+1];
        facts[0] = Self::one();
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Self{ facts: facts, ifacts: ifacts }
    }

    // nCk
    pub fn nk<T: Copy+Ord+Zero+IntoT<us>+Sub<Output=T>>(&self, n: T, k: T) -> Modint::<M> {
        if n < T::zero() || k < T::zero() || n < k { return Self::zero(); }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }

    // k!
    pub fn fact<T: Copy+IntoT<us>>(&self, k: T)  -> Modint<M> { self.facts[k.into_t()] }

    pub fn ifact<T: Copy+IntoT<us>>(&self, k: T) -> Modint<M> { self.ifacts[k.into_t()] }

    fn zero() -> Modint<M> { Modint::<M>::zero() }
    fn one() -> Modint<M> { Modint::<M>::one() }

}

