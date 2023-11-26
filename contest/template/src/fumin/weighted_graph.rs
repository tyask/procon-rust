#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use crate::common::*;

pub struct WeightedGraph<T> {
    g: Vec<Vec<(us, T)>>,
}

impl<T:Copy> WeightedGraph<T> {
    pub fn new(n: us) -> Self { Self { g: vec![vec![]; n] } }

    pub fn digraph(n: us, uvw: &Vec<(us,us,T)>) -> Self {
        let mut g = Self::new(n);
        uvw.iter().for_each(|&(u,v,w)|g.add(u,v,w));
        g
    }
    pub fn undigraph(n: us, uvw: &Vec<(us,us,T)>) -> Self {
        let mut g = Self::new(n);
        uvw.iter().for_each(|&(u,v,w)|g.add2(u,v,w));
        g
    }

    pub fn add(&mut self, u: us, v: us, w: T) { self[u].push((v, w)); }
    pub fn add2(&mut self, u: us, v: us, w: T) { self.add(u,v,w); self.add(v,u,w); }
}

impl<I: IntoT<us>, T> Index<I> for WeightedGraph<T> {
    type Output = Vec<(us, T)>;
    fn index(&self, i: I) -> &Self::Output { &self.g[i.into_t()] }
}

impl<I: IntoT<us>, T> IndexMut<I> for WeightedGraph<T> {
    fn index_mut(&mut self, i: I) -> &mut Self::Output { &mut self.g[i.into_t()] }
}
