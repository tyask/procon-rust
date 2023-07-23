#![allow(dead_code)]
use std::ops::Index;

use itertools::iproduct;
use crate::{common::*, chmin};

#[derive(Clone, Copy)]
pub struct Edge {
    pub a: us,
    pub b: us,
    pub cost: i64,
}

impl Edge {
    pub fn new(a: us, b: us, cost: i64) -> Self { Self { a, b, cost } }
    pub fn rev(&self) -> Self { Self::new(self.b, self.a, self.cost) }
}

impl From<(us, us, i64)> for Edge {
    fn from(t: (us, us, i64)) -> Self { Self::new(t.0, t.1, t.2) }
}

pub struct WarshallFloyd {
    pub dist: Vec<Vec<i64>>,
    pub prev: Vec<Vec<us>>,
}

impl WarshallFloyd {
    pub fn new(n: us) -> Self {
        Self { dist: vec![vec![i64::INF; n]; n], prev: vec![vec![0; n]; n] }
    }

    pub fn add(&mut self, e: Edge) {
        self.dist[e.a][e.b] = e.cost;
    }
    pub fn add_each(&mut self, e: Edge) {
        self.add(e);
        self.add(e.rev());
    }

    pub fn adds(&mut self, es: &Vec<Edge>) -> &mut Self {
        for &e in es { self.add(e); }
        self
    }

    pub fn adds_each(&mut self, es: &Vec<Edge>) -> &mut Self {
        for &e in es { self.add_each(e); }
        self
    }

    pub fn run(&mut self) -> &mut Self {
        let n = self.dist.len();
        let dist = &mut self.dist;
        let prev = &mut self.prev;
        prev.iter_mut().enumerate().for_each(|(i,v)|v.fill(i));
        for (k, i, j) in iproduct!(0..n, 0..n, 0..n) {
            if chmin!(dist[i][j], dist[i][k] + dist[k][j]) {
                prev[i][j] = prev[k][j];
            }
        }
        self
    }

    pub fn dist(&self, s: us, e: us) -> i64 {
        self.dist[s][e]
    }

    pub fn path(&self, s: us, e: us) -> Vec<us> {
        let mut p = vec![];
        let mut c = e;
        while c != s { p.push(c); c = self.prev[s][c]; }
        p.iter().chain(&[s]).rev().cloned().cv()
    }

    pub fn has_negative_cycle(&self) -> bool {
        (0..self.dist.len()).any(|i|self.dist(i, i) < 0)
    }
}

impl Index<(us, us)> for WarshallFloyd {
    type Output = i64;
    fn index(&self, index: (us, us)) -> &Self::Output { &self.dist[index.0][index.1] }
}

impl From<&Vec<Vec<i64>>> for WarshallFloyd {
    fn from(v: &Vec<Vec<i64>>) -> Self {
        let mut w = Self::new(v.len());
        for (i, j) in iproduct!(0..v.len(), 0..v.len()) { w.add(Edge::new(i, j, v[i][j])); }
        w
    }
}
