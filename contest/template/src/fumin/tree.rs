#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use crate::common::*;
use super::graph::Graph;

// CAP(fumin::graph)

#[derive(Debug, Clone)]
pub struct Tree(pub Graph);

impl Tree {
    pub fn new(n: us, uv: &Vec<(us, us)>) -> Self { Self(Graph::undigraph(n, uv)) }
    pub fn len(&self) -> us { self.0.len() }
    pub fn bfs(&self, s: us) -> Vec<us> { self.0.bfs(s) }
    pub fn depth(&self, s: us) -> Vec<us> { self.bfs(s) }
}

impl Tree {
    // サブツリーのノード数(自身を含む)を計算する
    pub fn sub(&self, s: us) -> Vec<us> {
        let mut sub = vec![1; self.len()];
        let mut q = deque::new();
        q.push_back((s, us::INF, true));
        while let Some((v, p, pre)) = q.pop_back() {
            if pre {
                q.push_back((v,p,false));
                for &u in &self[v] { if u!=p { q.push_back((u,v,true)); } }
            } else {
                if p != us::INF { sub[p] += sub[v]; }
            }
        }
        sub
    }
}

impl<T: IntoT<us>> Index<T> for Tree {
    type Output = Vec<us>;
    fn index(&self, i: T) -> &Self::Output { &self.0[i.into_t()] }
}
impl<T: IntoT<us>> IndexMut<T> for Tree {
    fn index_mut(&mut self, i: T) -> &mut Self::Output { &mut self.0[i.into_t()] }
}
