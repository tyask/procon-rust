#![allow(dead_code)]
use itertools::Itertools;
use superslice::Ext;
use crate::common::*;

pub struct Compress<T> {
    m: Vec<T>
}

impl<T:Clone + Copy + Ord> Compress<T> {
    pub fn new(v: &[T]) -> Self { v.into() }
    pub fn compress(&self, v: T) -> us { self.m.lower_bound(&v) }
    pub fn compress_list(&self, v: &[T]) -> Vec<us> { v.map(|&x|self.compress(x)) }
    pub fn decompress(&self, v: us) -> T { self.m[v] }
    pub fn decompress_list(&self) -> Vec<T> { self.m.clone() }
}

impl<T: Clone + Ord> From<Vec<T>> for Compress<T> {
    fn from(value: Vec<T>) -> Self {
        Self { m: value.iter().cloned().sorted().dedup().cv() }
    }
}

impl<T: Clone + Ord> From<&[T]> for Compress<T> {
    fn from(value: &[T]) -> Self { value.to_vec().into() }
}

impl<T: Clone + Ord> From<Vec<Vec<T>>> for Compress<T> {
    fn from(value: Vec<Vec<T>>) -> Self { value.iter().flatten().cloned().cv().into() }
}
