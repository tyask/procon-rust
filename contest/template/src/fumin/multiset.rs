#![allow(dead_code)]
use std::ops::RangeBounds;
use crate::common::*;

#[derive(Clone)]
struct MultiSet<V> {
    pub m: bmap<V, i64>,
}

impl<V:Clone+Copy+Ord> MultiSet<V> {
    pub fn new() -> Self { Self { m: bmap::new() } }

    pub fn insert(&mut self, v: V) -> bool {
        *self.m.or_def_mut(&v) += 1;
        return self.m[&v] == 1;
    }

    pub fn remove(&mut self, v: V) -> bool {
        *self.m.or_def_mut(&v) -= 1;
        if self.m[&v] <= 0 {
            self.m.remove(&v);
            return true;
        }
        return false;
    }

    pub fn range(&self, r: impl RangeBounds<V>) -> impl DoubleEndedIterator<Item=V> + '_ {
        self.m.range(r).map(|p|*p.0)
    }

    pub fn contains(&self, v: &V) -> bool {
        self.m.contains_key(&v)
    }
}

