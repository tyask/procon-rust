#![allow(dead_code)]
use std::{*, cmp::Reverse};
use crate::{common::*, chmin};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: us,
    pub to:   us,
    pub cost: i64
}
pub struct Dijkstra {
    pub g: Vec<Vec<Edge>>,
    pub dist: Vec<i64>,
    pub prev: Vec<us>,
}

impl Edge {
    pub fn new(from: us, to: us, cost: i64) -> Self { Self { from, to, cost } }
    pub fn rev(self) -> Self { let mut r = self.clone(); mem::swap(&mut r.from, &mut r.to); r }
}

impl Dijkstra {
    pub fn new(n: us) -> Self { Self { g: vec![vec![]; n], dist: vec![0;n], prev: vec![0;n] }}
    pub fn add(&mut self, e: Edge) -> &mut Self { self.g[e.from].push(e); self }
    pub fn add2(&mut self, e: Edge) -> &mut Self { self.add(e).add(e.rev()) }

    pub fn run(&mut self, s: us) -> &Vec<i64> {
        type P = (i64, us); // cost, node

        self.dist.fill(i64::INF);
        self.prev.fill(us::INF);

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

        dist
    }

    pub fn restore_shortest_path(&self, mut t: us) -> Vec<us> {
        let mut p = vec![];
        while t != us::INF { p.push(t); t = self.prev[t]; }
        p.reverse();
        p
    }

}
