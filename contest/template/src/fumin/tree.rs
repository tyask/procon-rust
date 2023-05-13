#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use crate::common::*;
use super::graph::Graph;

// CAP(fumin::graph)

pub struct Tree(Graph);

impl Tree {
    pub fn new(n: us, uv: &Vec<(us, us)>) -> Self {
        Self(Graph::undigraph(n, uv))
    }
    pub fn len(&self) -> us { self.0.len() }
    pub fn bfs(&self, s: us) -> Vec<us> { self.0.bfs(s) }
    pub fn depth(&self, s: us) -> Vec<us> {
        let mut dep = vec![0; self.len()];
        let mut que = deque::new();
        que.push_back((s, us::INF));
        while let Some((v, p)) = que.pop_back() {
            for &u in &self[v] { if u != p {
                dep[u] = dep[v]+1;
                que.push_back((u, v));
            }}
        }
        dep
    }
}
impl<T: IntoT<us>> Index<T> for Tree {
    type Output = Vec<us>;
    fn index(&self, i: T) -> &Self::Output { &self.0[i.into_t()] }
}
impl<T: IntoT<us>> IndexMut<T> for Tree {
    fn index_mut(&mut self, i: T) -> &mut Self::Output { &mut self.0[i.into_t()] }
}
