#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use crate::common::*;

// random access: O(1)
// insert: O(1)
// erase: O(1)
// check: O(1)
// clear: O(N)
// max value: fixed
pub struct FastVec<T, const N: us> {
    data: [T; N],
    len: us,
}

impl<T: Default+Clone+Copy, const N: us> Default for FastVec<T, N> {
    fn default() -> Self {
        Self {
            data: [T::default(); N],
            len: 0,
        }
    }
}

impl<T: Clone+Copy, const N: us> FastVec<T, N> {
    pub fn new(a: T) -> Self {
        Self {
            data: [a; N],
            len: 0,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=&T> { self.data.iter().take(self.len) }
    pub fn clear(&mut self) { self.len = 0; }
    pub fn len(&self) -> us { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn push(&mut self, a: T) {
        assert!(self.len < N);
        self.data[self.len] = a;
        self.len += 1;
    }

    pub fn remove(&mut self, index: us) {
        assert!(index < N);

        // indexはずれるので注意
        self.len -= 1;
        self.data[index] = self.data[self.len];
    }

}

impl<T, const N:us> Index<us> for FastVec<T, N> {
    type Output = T;
    fn index(&self, index: us) -> &Self::Output {
        assert!(index < self.len);
        &self.data[index]
    }
}

impl<T, const N:us> IndexMut<us> for FastVec<T, N> {
    fn index_mut(&mut self, index: us) -> &mut Self::Output {
        assert!(index < self.len);
        &mut self.data[index]
    }
}


// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut s = FastVec::<us, 10>::default();
        s.push(2);
        s.push(3);
        s.push(5);
        assert_eq!(s.len(), 3);
        assert_eq!(s.iter().cloned().cv(), vec![2, 3, 5]);

        s.remove(0);
        assert_eq!(s.len(), 2);
        assert_eq!(s.iter().cloned().cv(), vec![5, 3]);
    }
}