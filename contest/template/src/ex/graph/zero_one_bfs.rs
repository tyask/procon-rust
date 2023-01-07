#![allow(dead_code)]

pub struct ZeroOneBfs {
    pub g: Vec<Vec<(usize, isize)>>
}

impl ZeroOneBfs {
    pub fn new(n: usize) -> ZeroOneBfs { ZeroOneBfs { g: vec![vec![]; n] } }
    pub fn add_path(&mut self, u: usize, v: usize, cost: isize) {
        let g = &mut self.g;
        assert!(u < g.len() && v < g.len() && (cost == 0 || cost == 1));
        g[u].push((v,cost));
        g[v].push((u,cost));
    }

    pub fn run(&self, s: usize) -> Vec<isize> {
        let g = &self.g;
        assert!(s < g.len());
        let mut que = std::collections::VecDeque::new();
        let mut dist = vec![isize::MAX; g.len()];
        dist[s] = 0;
        que.push_back(s);
        while let Some(v) = que.pop_front() {
            for &(n, c) in &g[v] {
                let nc = dist[v] + c;
                if nc < dist[n] {
                    dist[n] = nc;
                    if c == 1 { que.push_back(n) }
                    else { que.push_front(n) }
                }
            }
        }
        dist
    }
}
