#![allow(dead_code)]
use std::*;
use crate::common::*;

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: us,
    pub to:   us,
    pub cost: i64
}

impl Edge {
    pub fn new(from: us, to: us, cost: i64) -> Self { Self { from, to, cost } }
    pub fn rev(self) -> Self { let mut r = self.clone(); mem::swap(&mut r.from, &mut r.to); r }
}

pub struct BellmanFord {
    pub edges: Vec<Edge>,
    pub dist: Vec<i64>,
    pub prev: Vec<us>,
}

impl BellmanFord {
    pub fn new(n: us) -> Self { Self { edges: vec![], dist: vec![0;n], prev: vec![0;n] }}
    pub fn add(&mut self, e: Edge) -> &mut Self { self.edges.push(e); self }
    pub fn add2(&mut self, e: Edge) -> &mut Self { self.add(e).add(e.rev()) }

    pub fn run(&mut self, s: us) -> &Vec<i64> {
        self.dist.fill(i64::INF);
        self.prev.fill(us::INF);

        let dist = &mut self.dist;
        let prev = &mut self.prev;
        let n = dist.len();

        dist[s] = 0;
        let mut cnt = 0;
        while cnt < n {
            let mut updated = false;
            for &e in &self.edges {
                let cost = dist[e.from] + e.cost;
                if dist[e.from] != i64::INF && cost < dist[e.to] {
                    dist[e.to] = cost;
                    prev[e.to] = e.from;
                    updated = true;
                }
            }

            if !updated { break; }
            cnt += 1;
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
