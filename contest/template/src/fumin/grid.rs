#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use itertools::iproduct;

use crate::common::*;
use super::pt::Pt;

// CAP(fumin::pt)

pub struct Grid<T>(pub Vec<Vec<T>>);

impl<T: Clone> Grid<T> {
    pub fn new(h: us, w: us, v: T) -> Self { Self::from(&vec![vec![v; w]; h]) }
}
impl<T> Grid<T> {
    pub fn h(&self) -> us { self.0.len() }
    pub fn w(&self) -> us { self.0[0].len() }
    pub fn is_in_p(&self, p: Pt<us>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t(&self, t: (us,us)) -> bool { t.0 < self.h() && t.1 < self.w() }
}
impl<T: Clone+Eq> Grid<T> {
    pub fn position(&self, t: &T) -> Option<Pt<us>> {
        iproduct!(0..self.h(), 0..self.w()).into_iter().map(|(i,j)|Pt::<us>::new(i,j)).filter(|&p|self[p]==*t).next()
    }
}
impl<T: Clone> From<&Vec<Vec<T>>> for Grid<T> {
    fn from(v: &Vec<Vec<T>>) -> Self { Self(v.to_vec()) }
}
impl<T, N: IntoT<us>> Index<Pt<N>> for Grid<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: IntoT<us>> IndexMut<Pt<N>> for Grid<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T, N: IntoT<us>> Index<(N,N)> for Grid<T> {
    type Output = T;
    fn index(&self, p: (N,N)) -> &Self::Output { &self.0[p.0.us()][p.1.us()] }
}
impl<T, N: IntoT<us>> IndexMut<(N,N)> for Grid<T> {
    fn index_mut(&mut self, p: (N,N)) -> &mut Self::Output { &mut self.0[p.0.us()][p.1.us()] }
}
impl<T, N: IntoT<us>> Index<N> for Grid<T> {
    type Output = Vec<T>;
    fn index(&self, p: N) -> &Self::Output { &self.0[p.us()] }
}
impl<T, N: IntoT<us>> IndexMut<N> for Grid<T> {
    fn index_mut(&mut self, p: N) -> &mut Self::Output { &mut self.0[p.us()] }
}
impl Grid<char> {
    pub fn bfs(&self, s: Pt<us>) -> Grid<us> {
        let mut que = deque::new();
        let mut m = Grid::new(self.h(), self.w(), us::INF);
        que.push_back(s);
        m[s] = 0;
        while let Some(v) = que.pop_front() {
            for d in Pt::<us>::D4 {
                let nv = v.wrapping_add(d);
                if self.is_in_p(nv) && self[nv]!='#' && m[nv]==us::INF {
                    m[nv] = m[v]+1;
                    que.push_back(nv);
                }
            }
        }
        m
    }
}
