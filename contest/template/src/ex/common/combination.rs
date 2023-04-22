#![allow(dead_code)]
use num::{Zero, One};

use crate::common::*;
use super::modint::{Modint, Modules};

// CAP(ex::common::modint)

struct Combination<M> {
    facts: Vec<Modint<M>>,
    ifacts: Vec<Modint<M>>
}

impl<M: Modules> Combination<M> {
    pub fn new(n: us) -> Combination<M> {
        assert!(n.i64() < M::MOD);
        let mut facts  = vec![Modint::<M>::zero(); n+1];
        let mut ifacts = vec![Modint::<M>::zero(); n+1];
        facts[0] = Modint::<M>::one();
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Combination{facts: facts, ifacts: ifacts}
    }

    // nCk
    pub fn comb(&self, n: us, k: us) -> Modint<M> {
        assert!(n.i64() < M::MOD);
        if n < k { return Modint::<M>::zero(); }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }
    // k!
    pub fn fact(&self, k: us)  -> Modint<M> { self.facts[k] }
    pub fn ifact(&self, k: us) -> Modint<M> { self.ifacts[k] }
}