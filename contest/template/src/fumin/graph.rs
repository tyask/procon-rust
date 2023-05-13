#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use crate::common::*;

#[derive(Clone,Debug)]
pub struct Graph(pub Vec<Vec<us>>);

impl Graph {
    pub fn digraph(n: us, uv: &Vec<(us, us)>) -> Self {
        let mut g = vec![vec![]; n];
        uv.iter().for_each(|&(u,v)|g[u].push(v));
        Self(g)
    }
    pub fn undigraph(n: us, uv: &Vec<(us, us)>) -> Self {
        let mut g = vec![vec![]; n];
        uv.iter().for_each(|&(u,v)|{g[u].push(v); g[v].push(u);});
        Self(g)
    }
    pub fn len(&self) -> us { self.0.len() }
    pub fn rev(&self) -> Self {
        let ruv = self.0.iter().enumerate()
            .flat_map(|(i,v)|v.iter().map(move |&j|(j,i))).cv();
        Self::digraph(self.len(), &ruv)
    }
    pub fn bfs(&self, s: us) -> Vec<us> {
        let mut d = vec![us::INF; self.len()];
        d[s] = 0;
        let mut que = deque::new();
        que.push_back(s);
        while let Some(a) = que.pop_front() {
            for &b in &self[a] {
                if d[b] > d[a] + 1 {
                    d[b] = d[a] + 1;
                    que.push_back(b);
                }
            }
        }
        d
    }
}
impl<T: IntoT<us>> Index<T> for Graph {
    type Output = Vec<us>;
    fn index(&self, i: T) -> &Self::Output { &self.0[i.into_t()] }
}
impl<T: IntoT<us>> IndexMut<T> for Graph {
    fn index_mut(&mut self, i: T) -> &mut Self::Output { &mut self.0[i.into_t()] }
}
