#![allow(dead_code)]

use crate::common::Unit;
use super::modint::{Modint, Modules};

// CAP(ex::modint)

struct Combination<M> { facts: Vec<Modint<M>>, ifacts: Vec<Modint<M>> }

impl<M: Modules> Combination<M> {
    pub fn new(n: usize) -> Combination<M> {
        assert!(n < M::MOD);
        let mut facts  = vec![Modint::<M>::ZERO; n+1];
        let mut ifacts = vec![Modint::<M>::ZERO; n+1];
        facts[0] = Modint::<M>::ONE;
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Combination{facts: facts, ifacts: ifacts}
    }

    // nCk
    pub fn comb(&self, n: usize, k: usize) -> Modint<M> {
        assert!(n < M::MOD);
        if n < k { return Modint::<M>::ZERO; }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }
    // k!
    pub fn fact(&self, k: usize)  -> Modint<M> { self.facts[k] }
    pub fn ifact(&self, k: usize) -> Modint<M> { self.ifacts[k] }
}