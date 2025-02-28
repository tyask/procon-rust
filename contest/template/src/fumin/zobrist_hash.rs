#![allow(non_camel_case_types)]
use std::hash::BuildHasherDefault;
use itertools::iproduct;
use rustc_hash::FxHasher;

use crate::common::us;

type map<K,V>  = std::collections::HashMap<K,V, BuildHasherDefault<FxHasher>>;
type P = super::pt::Pt<us>;
// CAP(fumin::pt)

#[derive(Debug, Clone)]
pub struct ZobristHash<T: std::hash::Hash + std::cmp::Eq + Clone> {
    hash: map<T, u64>,
}

impl<T: std::hash::Hash + std::cmp::Eq + Clone> ZobristHash<T> {
    pub fn new(items: &[T], rng: &mut impl rand_core::RngCore) -> ZobristHash<T> {
        let mut hash = map::default();
        for item in items { hash.insert(item.clone(), rng.next_u64()); }
        ZobristHash { hash }
    }

    pub fn hash(&self, x:&T) -> u64 { self.hash[&x] }
    pub fn hash_s(&self, v:&[T]) -> u64 { v.iter().map(|x|self.hash(x)).fold(0,|a,x|a^x) }
}

#[derive(Debug, Clone)]
pub struct ZobristHash2d {
    hash: Vec<Vec<u64>>,
}

impl ZobristHash2d {
    pub fn new(h:usize, w:usize, rng: &mut impl rand_core::RngCore) -> Self {
        let mut hash = vec![vec![0; w]; h];
        for (i, j) in iproduct!(0..h, 0..w) { hash[i][j] = rng.next_u64(); }
        Self { hash }
    }

    pub fn hash(&self, v:P) -> u64 { self.hash[v.x][v.y] }
    pub fn hash_s(&self, v:&[P]) -> u64 { v.iter().map(|&x|self.hash(x)).fold(0,|a,x|a^x) }
}
