#![allow(dead_code)]

struct FastBitSet {
    bs: Vec<u32>,
    id: u32,
}

impl FastBitSet {
    fn new(n: usize) -> Self { Self { bs: vec![0; n], id: 1, } }
    fn clear(&mut self) { self.id += 1; }
    fn set(&mut self, i: usize, f: bool) { self.bs[i] = if f { self.id } else { 0 }; }
}

impl std::ops::Index<usize> for FastBitSet {
    type Output = bool;
    fn index(&self, i: usize) -> &bool {
        if self.bs[i] == self.id { &true } else { &false }
    }
}
