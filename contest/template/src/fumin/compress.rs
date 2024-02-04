#![allow(dead_code)]
use itertools::Itertools;
use superslice::Ext;
use crate::common::*;

pub struct Compress<T> {
    m: Vec<T>
}

impl<T:Clone + Ord> Compress<T> {
    pub fn new(v: &[T]) -> Self {
        Self { m: v.iter().cloned().sorted().dedup().cv() }
    }
    pub fn new2d(v: &[Vec<T>]) -> Self {
        Self { m: v.iter().flatten().cloned().sorted().dedup().cv() }
    }
    pub fn compress(&self, v: &T) -> us { self.m.lower_bound(v) }
    pub fn compress_list(&self, v: &[T]) -> Vec<us> { v.map(|x|self.compress(x)) }
    pub fn decompress(&self, v: us) -> &T { &self.m[v] }
    pub fn decompress_list(&self) -> Vec<T> { self.m.clone() }
    pub fn len(&self) -> us { self.m.len() }
}

impl<T: Clone + Ord> From<Vec<T>> for Compress<T> {
    fn from(value: Vec<T>) -> Self { Self::new(&value) }
}

impl<T: Clone + Ord> From<&[T]> for Compress<T> {
    fn from(value: &[T]) -> Self { value.to_vec().into() }
}

impl<T: Clone + Ord> From<Vec<Vec<T>>> for Compress<T> {
    fn from(value: Vec<Vec<T>>) -> Self { value.iter().flatten().cloned().cv().into() }
}
