#![allow(dead_code)]
use std::ops::Index;

use crate::common::*;

// random access: O(1)
// insert: O(1)
// erase: O(1)
// check: O(1)
// clear: O(N)
// max value: fixed
pub struct FastISet<T, const N: us> {
    data: [T; N],
    indices: [us; N],
    len: us,
}

impl<T: Default+Clone+Copy, const N: us> Default for FastISet<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
            indices: [us::INF; N],
            len: 0,
        }
    }
}

impl<T: Clone+Copy+IntoT<us>, const N: us> FastISet<T, N> {

    pub fn iter(&self) -> impl Iterator<Item=&T> { self.data.iter().take(self.len) }
    pub fn len(&self) -> us { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn contains(&mut self, a: &T) -> bool { self.indices[(*a).into_t()] != us::INF }
    pub fn clear(&mut self) { self.indices.fill(us::INF); }

    pub fn insert(&mut self, a: T) -> bool {
        let a_us: us = a.into_t();
        assert!(a_us < N);
        if self.indices[a_us] != us::INF { return false; }
        self.data[self.len] = a;
        self.indices[a_us] = self.len;
        self.len += 1;
        true
    }

    pub fn remove(&mut self, a: &T) -> bool {
        let a_us: us = (*a).into_t();
        let index = self.indices[a_us];
        if index == us::INF { return false; }
        assert!(self.len > 0);

        // 先頭num個が有効な値となるように、最後の要素をdata[num]に移動する
        self.len -= 1;
        self.data[index] = self.data[self.len];
        self.indices[self.data[self.len].into_t()] = index;
        self.indices[a_us] = us::INF;
        true
    }


}

impl<T, const N:us> Index<us> for FastISet<T, N> {
    type Output = T;

    fn index(&self, index: us) -> &Self::Output {
        assert!(index < self.len);
        &self.data[index]
    }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut s = FastISet::<u32, 10>::default();
        assert_eq!(s.insert(2), true);
        assert_eq!(s.insert(5), true);
        assert_eq!(s.len(), 2);
        assert_eq!(s.iter().cloned().cv(), vec![2, 5]);
        assert_eq!(s.contains(&2), true);
        assert_eq!(s.contains(&5), true);
        assert_eq!(s.contains(&3), false);

        assert_eq!(s.remove(&2), true);
        assert_eq!(s.len(), 1);
        assert_eq!(s.iter().cloned().cv(), vec![5]);
        assert_eq!(s.contains(&2), false);
        assert_eq!(s.contains(&5), true);
        assert_eq!(s.contains(&3), false);

    }
}