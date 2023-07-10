#![allow(dead_code)]
use crate::common::*;

// 昇順のヒープ

pub type rbheap = RevBinaryHeap;

pub struct RevBinaryHeap<T> {
    q: bheap<Reverse<T>>,
}

impl<T:Ord> RevBinaryHeap<T> {
    pub fn new() -> Self { Self { q: bheap::new() } }

    pub fn push(&mut self, t: T) {
        self.q.push(Reverse(t));
    }

    pub fn pop(&mut self) -> Option<T> {
        self.q.pop().map(|r|r.0)
    }

    pub fn peek(&mut self) -> Option<&T> {
        self.q.peek().map(|r|&r.0)
    }
}