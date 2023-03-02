#![allow(dead_code)]
use std::{*, cmp::Reverse};
use crate::{common::*, chmin};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    from: us,
    to:   us,
    cost: is
}
pub struct Dijkstra {
    g: Vec<Vec<Edge>>,
    dist: Vec<is>,
    prev: Vec<us>,
}

impl Edge {
    pub fn new(from: us, to: us, cost: is) -> Edge { Edge { from: from, to: to, cost: cost } }
    pub fn reverse(self) -> Edge { Self::new(self.to, self.from, self.cost) }
}

impl Dijkstra {
    pub fn new(n: us) -> Self { Self { g: vec![vec![]; n], dist: vec![0;n], prev: vec![0;n] }}
    pub fn add(&mut self, e: Edge) -> &mut Self { self.g[e.from].push(e); self }
    pub fn add2(&mut self, e: Edge) -> &mut Self { self.add(e).add(e.reverse()) }

    pub fn run(&mut self, s: us) {
        type P = (is, us); // cost, node

        self.dist = vec![is::INF; self.dist.len()];
        self.prev = vec![us::INF; self.prev.len()];

        let g = &self.g;
        let dist = &mut self.dist;
        let prev = &mut self.prev;
        let mut que = bheap::new();

        dist[s] = 0;
        que.push(Reverse((0, s)));
        while let Some(Reverse((cost, v))) = que.pop() {
            if dist[v] < cost { continue }
            for e in &g[v] {
                let nc = cost + e.cost;
                if chmin!(dist[e.to], nc) {
                    que.push(Reverse((dist[e.to], e.to)));
                    prev[e.to] = v;
                }
            }
        }
    }

    pub fn dist(&self, t: us) -> is {
        self.dist[t]
    }

    pub fn restore_shortest_path(&self, mut t: us) -> Vec<us> {
        let mut p = vec![];
        while t != us::INF { p.push(t); t = self.prev[t]; }
        p.reverse();
        p
    }

}
