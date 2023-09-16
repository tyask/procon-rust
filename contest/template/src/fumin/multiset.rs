#![allow(dead_code)]
use std::ops::RangeBounds;
use crate::common::*;

#[derive(Clone, Debug)]
pub struct MultiSet<V>(pub bmap<V, us>);

impl<V:Clone+Copy+Ord> MultiSet<V> {
    pub fn new() -> Self { Self(bmap::new()) }

    pub fn insert(&mut self, v: V) -> us {
        let c = self.0.or_def_mut(&v);
        *c += 1;
        *c
    }

    pub fn remove(&mut self, v: &V) -> bool {
        self.remove_n(v, 1)
    }

    pub fn remove_n(&mut self, v: &V, n: us) -> bool {
        if let Some(p) = self.0.get_mut(v) {
            if *p > 0 { *p -= us::min(*p, n); }
            if *p == 0 {
                self.0.remove(v);
                return true;
            }
        }
        return false;
    }

    pub fn count(&self, v: &V) -> us {
        self.0.or_def(v)
    }

    pub fn iter(&self) -> impl Iterator<Item=&V> {
        self.0.keys()
    }

    pub fn iter_counts(&self) -> impl Iterator<Item=(&V, us)> {
        self.0.iter().map(|p|(p.0,*p.1))
    }

    pub fn range(&self, r: impl RangeBounds<V>) -> impl DoubleEndedIterator<Item=(&V, us)> + '_ {
        self.0.range(r).map(|p|(p.0,*p.1))
    }

    pub fn range_values(&self, r: impl RangeBounds<V>) -> impl DoubleEndedIterator<Item=&V> + '_ {
        self.0.range(r).map(|p|p.0)
    }

    pub fn range_flatten(&self, r: impl RangeBounds<V>) -> impl DoubleEndedIterator<Item=(&V, us)> + '_ {
        self.0.range(r).flat_map(|p|(0..*p.1).into_iter().map(move|i|(p.0,i)))
    }

    pub fn contains(&self, v: &V) -> bool {
        self.0.contains_key(&v)
    }
}

