
#![allow(dead_code)]
use std::{*, cmp::Reverse};

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    from: usize,
    to:   usize,
    cost: isize
}
pub struct Dijkstra {
    g: Vec<Vec<Edge>>,
    dist: Vec<isize>,
    prev: Vec<usize>,
}

impl Edge {
    pub fn new(from: usize, to: usize, cost: isize) -> Edge { Edge { from: from, to: to, cost: cost } }
    pub fn reverse(self) -> Edge { Self::new(self.to, self.from, self.cost) }
}

impl Dijkstra {
    pub fn new(n: usize) -> Dijkstra { Dijkstra { g: vec![vec![]; n], dist: vec![0;n], prev: vec![0;n] }}
    pub fn add(&mut self, e: Edge) -> &mut Dijkstra { self.g[e.from].push(e); self }
    pub fn add2(&mut self, e: Edge) -> &mut Dijkstra { self.add(e).add(e.reverse()) }

    pub fn run(&mut self, s: usize) {
        type P = (isize, usize); // cost, node

        self.dist = vec![isize::MAX; self.dist.len()];
        self.prev = vec![usize::MAX; self.prev.len()];

        let g = &self.g;
        let dist = &mut self.dist;
        let prev = &mut self.prev;
        let mut que = std::collections::BinaryHeap::new();

        dist[s] = 0;
        que.push(Reverse((0, s)));
        while let Some(Reverse((cost, v))) = que.pop() {
            if dist[v] < cost { continue }
            for e in &g[v] {
                let nc = cost + e.cost;
                if Self::chmin(&mut dist[e.to], nc) {
                    que.push(Reverse((dist[e.to], e.to)));
                    prev[e.to] = v;
                }
            }
        }
    }

    pub fn dist(&self, t: usize) -> isize {
        self.dist[t]
    }

    pub fn restore_shortest_path(&self, mut t: usize) -> Vec<usize> {
        let mut p = vec![];
        while t != usize::MAX { p.push(t); t = self.prev[t]; }
        p.reverse();
        return p;
    }

    fn chmin(target: &mut isize, value: isize) -> bool { if *target > value { *target = value; true } else { false } }
}
