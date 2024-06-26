#![allow(dead_code)]
use crate::common::*;

use super::unionfind::Unionfind;

// CAP(fumin::unionfind)

#[derive(Clone,Copy,PartialEq,Eq,Hash)]
pub struct Edge { pub u: us, pub v: us, pub cost: i64 }
impl Edge { pub fn new(u: us, v: us, cost: i64) -> Self { Self { u, v, cost } } }

pub struct Kruskal {
    n: usize,
    edges: Vec<Edge>,
}

impl Kruskal {
    pub fn new(n: us) -> Self {
        assert!(n > 0);
        Self { n, edges: vec![] }
    }
    pub fn add_edge(&mut self, e: Edge) { self.edges.push(e); }
    pub fn add(&mut self, u: us, v: us, cost: i64) { self.edges.push(Edge::new(u, v, cost)); }

    pub fn run(&mut self) -> Option<Vec<Edge>> {
        self.edges.sort_by_key(|e|e.cost);
        let n = self.n;
        let mut uf = Unionfind::new(n);
        let mut ret = vec![];
        for &e in &self.edges { if uf.unite(e.u, e.v) { ret.push(e); }}
        if uf.size(0) == n { Some(ret) } else { None }
    }
}
