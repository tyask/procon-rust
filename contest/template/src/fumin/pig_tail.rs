#![allow(dead_code)]
use std::ops::Index;
use crate::common::*;

// 以下のように途中でループが始まる有効グラフのi番目の要素を取得する
// 3 > 5 > 2 > 1 > 7 > 6
//             ^       v
//             9 < 4 < 8
#[derive(Debug)]
struct PigTail<T> {
    v: Vec<T>,
    loop_start: us,
}
impl<T: Clone> PigTail<T> {
    // v: 有効グラフの要素を順番に並べたリスト
    // loop_start: ループが開始するindex
    pub fn new(v: &Vec<T>, loop_start: us) -> Self {
        Self { v: v.to_vec(), loop_start }
    }
    pub fn len(&self) -> us { self.v.len() }
}

impl PigTail<us> {
    // g: g[i]:=iの向き先
    // st: グラフの開始位置
    pub fn from_digraph(g: &Vec<us>, st: us) -> Self {
        let mut v = vec![];
        let mut s = set::new();
        let mut p = st;
        while s.insert(p) { v.push(p); p = g[p]; }
        Self::new(&v, v.pos(&p).unwrap())
    }
}

impl<T: Clone> Index<us> for PigTail<T> {
    type Output = T;
    fn index(&self, index: us) -> &Self::Output {
        let loop_start = self.loop_start;
        let loop_cnt = self.len() - loop_start;
        let i = if index < self.len() { index } else { loop_start + (index - loop_start) % loop_cnt };
        &self.v[i]
    }
}
