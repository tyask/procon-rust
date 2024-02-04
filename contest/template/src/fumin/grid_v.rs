#![allow(dead_code)]
use std::{ops::{Index, IndexMut}, cmp::Reverse};

use crate::{common::*, chmin};
use super::pt::{Pt, Dir};

// CAP(fumin::pt)

pub struct GridV<T> {
    g: Vec<T>,
    h: us,
    w: us,
}

impl <T> GridV<T>
where
    T: Clone + Default {
    pub fn new(h: us, w: us) -> Self {
        Self { g: vec![T::default(); h * w], h, w, }
    }
    pub fn with_default(h: us, w: us, v: T) -> Self {
        Self { g: vec![v; h * w], h, w, }
    }
    pub fn is_in_p<N: IntoT<us>>(&self, p: Pt<N>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t<N: IntoT<us>>(&self, t: (N, N)) -> bool { t.0.into_t() < self.h && t.1.into_t() < self.w }
}

impl<T, N: IntoT<us>> Index<N> for GridV<T> {
    type Output = [T];
    fn index(&self, i: N) -> &Self::Output {
        let idx = i.into_t() * self.h;
        &self.g[idx..idx+self.w]
    }
}
impl<T, N: IntoT<us>> IndexMut<N> for GridV<T> {
    fn index_mut(&mut self, i: N) -> &mut Self::Output {
        let idx = i.into_t() * self.h;
        &mut self.g[idx..idx+self.w]
    }
}

impl<T, N: IntoT<us>> Index<(N,N)> for GridV<T> {
    type Output = T;
    fn index(&self, index: (N,N)) -> &Self::Output { &self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> IndexMut<(N,N)> for GridV<T> {
    fn index_mut(&mut self, index: (N,N)) -> &mut Self::Output { &mut self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> Index<Pt<N>> for GridV<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: IntoT<us>> IndexMut<Pt<N>> for GridV<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T: Clone> From<&Vec<Vec<T>>> for GridV<T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        let (h, w) = (value.len(), value[0].len());
        GridV{ g: value.iter().cloned().flatten().cv(), h, w }
    }
}

pub struct ShortestPath {
    pub start: Pt<us>,
    pub dist: GridV<i64>,
    pub prev: GridV<Pt<us>>,
}

impl ShortestPath {
    pub fn restore_shortest_path(&self, mut t: Pt<us>) -> Vec<Pt<us>> {
        let mut ps = vec![];
        while t != Pt::<us>::INF { ps.push(t); t = self.prev[t]; }
        ps.reverse();
        assert!(ps[0] == self.start);
        ps
    }
}

impl GridV<char> {
    // まだあまり動かしてないので、そのうちテスト必要
    pub fn bfs(&self, s: Pt<us>) -> ShortestPath {
        let mut que = deque::new();
        let mut ret = ShortestPath {
            start: s,
            dist: GridV::with_default(self.h, self.w, i64::INF),
            prev: GridV::with_default(self.h, self.w, Pt::<us>::INF),
        };
        que.push_back(s);
        ret.dist[s] = 0;
        while let Some(v) = que.pop_front() {
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if self.is_in_p(nv) && self[nv]!='#' && ret.dist[nv]==i64::INF {
                    ret.dist[nv] = ret.dist[v]+1;
                    ret.prev[nv] = v;
                    que.push_back(nv);
                }
            }
        }
        ret
    }
}


pub trait CellTrait {
    fn cost(&self, d: Dir) -> Option<i64>;
}

impl<T: CellTrait> GridV<T> {
    pub fn dijkstra(&self, s: Pt<us>) -> ShortestPath {
        type P = Pt<us>;
        let mut ret = ShortestPath {
            start: s,
            dist: GridV::with_default(self.h,self.w,i64::INF),
            prev: GridV::with_default(self.h,self.w,P::INF),
        };
        let mut q = bheap::new();
        q.push(Reverse((0,s)));
        ret.dist[s] = 0;
        while let Some(Reverse((cost, v))) = q.pop() {
            if ret.dist[v] < cost { continue; }
            for d in Dir::VAL4 {
                let Some(c) = self[v].cost(d) else { continue; };
                let nv = v.next(d);
                let nc = cost + c;
                if chmin!(ret.dist[nv], nc) {
                    q.push(Reverse((nc, nv)));
                    ret.prev[nv] = v;
                }
            }
        }
        ret
    }
}
