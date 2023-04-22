#![allow(dead_code)]
use crate::{common::*, chmin};

pub struct ZeroOneBfs {
    pub g: Vec<Vec<(us, is)>>
}

impl ZeroOneBfs {
    pub fn new(n: us) -> Self { Self { g: vec![vec![]; n] } }

    pub fn add(&mut self, u: us, v: us, cost: is) {
        let g = &mut self.g;
        assert!(u < g.len() && v < g.len() && (cost == 0 || cost == 1));
        g[u].push((v,cost));
    }

    pub fn add2(&mut self, u: us, v: us, cost: is) {
        self.add(u, v, cost);
        self.add(v, u, cost);
    }

    pub fn run(&self, s: us) -> Vec<is> {
        let g = &self.g;
        assert!(s < g.len());
        let mut que = deque::new();
        let mut dist = vec![is::INF; g.len()];
        dist[s] = 0;
        que.push_back(s);
        while let Some(v) = que.pop_front() {
            for &(n, c) in &g[v] {
                if chmin!(dist[n], dist[v]+c) {
                    if c == 0 { que.push_front(n); } else { que.push_back(n); }
                }
            }
        }
        dist
    }
}
