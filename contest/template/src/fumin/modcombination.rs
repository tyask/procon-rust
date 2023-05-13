#![allow(dead_code)]
use std::ops::Sub;
use num::{Zero, One};
use crate::common::*;
use super::modint::{Modint, Modules};

// CAP(fumin::modint)

pub struct ModCombination<M> {
    facts: Vec<M>,
    ifacts: Vec<M>
}

impl<Mod: Modules> ModCombination<Modint<Mod>> {
    pub fn new(n: us) -> Self {
        assert!(n.i64() < Modint::<Mod>::MOD);
        let mut facts  = vec![Self::zero(); n+1];
        let mut ifacts = vec![Self::zero(); n+1];
        facts[0] = Self::one();
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Self{ facts: facts, ifacts: ifacts }
    }

    // nCk
    pub fn nk<T: Copy+Ord+Zero+IntoT<us>+Sub<Output=T>>(&self, n: T, k: T) -> Modint::<Mod> {
        if n < T::zero() || k < T::zero() || n < k { return Self::zero(); }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }

    // k!
    pub fn fact<T: Copy+IntoT<us>>(&self, k: T)  -> Modint<Mod> { self.facts[k.into_t()] }

    pub fn ifact<T: Copy+IntoT<us>>(&self, k: T) -> Modint<Mod> { self.ifacts[k.into_t()] }

    fn zero() -> Modint<Mod> { Modint::<Mod>::zero() }
    fn one() -> Modint<Mod> { Modint::<Mod>::one() }

}

