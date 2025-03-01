#![allow(dead_code)]

use crate::common::us;
use super::pt;

// CAP(fumin::pt)
type P = pt::Pt<us>;

// フラグのチェックとclearがO(1)でできる
pub struct FastBitSet {
    bs: Vec<u32>,
    id: u32,
}

impl FastBitSet {
    pub fn new(n: usize) -> Self { Self { bs: vec![0; n], id: 1, } }
    pub fn clear(&mut self) { self.id += 1; }
    pub fn set(&mut self, i: usize, f: bool) { self.bs[i] = if f { self.id } else { 0 }; }
}

impl std::ops::Index<usize> for FastBitSet {
    type Output = bool;
    fn index(&self, i: usize) -> &bool {
        if self.bs[i] == self.id { &true } else { &false }
    }
}

pub struct FastBitSet2d {
    bs: Vec<u32>,
    id: u32,
    pub h: us,
    pub w: us,
}

impl FastBitSet2d {
    pub fn new(h: us, w:us) -> Self { Self { bs: vec![0; h*w], id: 1, h, w, } }
    pub fn clear(&mut self) { self.id += 1; }
    pub fn set(&mut self, v: P, f: bool) {
        let idx = self.idx(v);
        self.bs[idx] = if f { self.id } else { 0 };
    }
    fn idx(&self, v: P) -> us { v.x * self.w + v.y }
}

impl std::ops::Index<P> for FastBitSet2d {
    type Output = bool;
    fn index(&self, v: P) -> &bool {
        if self.bs[self.idx(v)] == self.id { &true } else { &false }
    }
}

