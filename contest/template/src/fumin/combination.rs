#![allow(dead_code)]
use num::{Zero, One};
use crate::common::*;
use super::modint::{Modint, Modules};

// CAP(fumin::modint)

pub struct Combination<M> {
    facts: Vec<M>,
    ifacts: Vec<M>
}

impl<Mod: Modules> Combination<Modint<Mod>> {
    pub fn new(n: us) -> Self {
        assert!(n.i64() < Modint::<Mod>::MOD);
        let mut facts  = vec![Modint::<Mod>::zero(); n+1];
        let mut ifacts = vec![Modint::<Mod>::zero(); n+1];
        facts[0] = Modint::<Mod>::one();
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Self{ facts: facts, ifacts: ifacts }
    }

    // nCk
    pub fn comb(&self, n: us, k: us) -> Modint::<Mod> {
        if n < k { return Modint::<Mod>::zero(); }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }

    // k!
    pub fn fact(&self, k: us)  -> Modint<Mod> { self.facts[k] }

    pub fn ifact(&self, k: us) -> Modint<Mod> { self.ifacts[k] }
}
