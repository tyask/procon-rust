
#[allow(dead_code)]
pub mod multiset {
use std::{collections::BTreeSet, ops::{RangeBounds, Bound}};

struct MultiSet<V> {
    m: BTreeSet<(V, usize)>,
    uniq: usize,
}

impl<V:Clone+Copy+Ord> MultiSet<V> {
    pub fn new() -> MultiSet<V> { MultiSet{m: BTreeSet::new(), uniq: 0 }}
    pub fn insert(&mut self, v: V) -> bool { self.uniq += 1; self.m.insert((v, self.uniq)) }
    pub fn range<R: RangeBounds<V>>(&self, r: R) -> impl DoubleEndedIterator<Item=V> + '_ {
        let s = match r.start_bound() {
            Bound::Included(&x) => Bound::Included((x, std::usize::MIN)),
            Bound::Excluded(&x) => Bound::Excluded((x, std::usize::MIN)),
            Bound::Unbounded    => Bound::Unbounded
        };
        let e = match r.end_bound() {
            Bound::Included(&x) => Bound::Included((x, std::usize::MAX)),
            Bound::Excluded(&x) => Bound::Excluded((x, std::usize::MAX)),
            Bound::Unbounded    => Bound::Unbounded
        };
        self.m.range((s, e)).map(|p|p.0)
    }
    pub fn contains(&self, v: &V) -> bool { self.range(v..=v).next().is_some() }
}

}

