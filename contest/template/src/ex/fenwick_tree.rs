
#[allow(dead_code)]
mod fenwick_tree {
use std::*;

// 1-indexed
struct FenwickTree { n: usize, dat: Vec<isize> }
impl FenwickTree {
    pub fn new(n: usize) -> FenwickTree { FenwickTree{ n: n, dat: vec![0; n] } }

    // O(logN)
    pub fn add(&mut self, mut p: usize, v: isize) {
        assert! {p < self.n}
        while p <= self.n { self.dat[p] += v; p += Self::lsb(p); }
    }
    pub fn set(&mut self, p: usize, v: isize) { self.add(p, v-self.get(p)) }

    // sum of [0, p) O(logN)
    pub fn sum(&self, mut p: usize) -> isize {
        assert! {p < self.n}
        let mut r = 0;
        while p > 0 { r += self.dat[p]; p -= Self::lsb(p); }
        r
    }
    // sum of [l, r) O(logN)
    pub fn sum2(&self, l: usize, r: usize) -> isize { self.sum(r) - self.sum(l) }
    pub fn get(&self, p: usize) -> isize { self.sum2(p, p+1) }

    pub fn lower_bound(&self, mut v: isize) -> usize {
        if v <= 0 { return 0; }
        let (mut x, mut k) = (0, Self::largest_pow2(self.n));
        while k > 0 {
            if x+k <= self.n && self.dat[x+k-1] < v { v -= self.dat[x+k-1]; x += k; }
            k /= 2;
        }
        x
    }

    // least significant bit (引数を割り切る最大の2冪)
    fn lsb(n: usize) -> usize { let n = n as isize; (n & -n) as usize }
    fn largest_pow2(n: usize) -> usize { let mut r = 1; while r * 2 <= n { r *= 2; } r }
}

// 転倒数
pub fn count_inversions(v: Vec<usize>) -> usize {
    let mut r = 0;
    let mut t = FenwickTree::new(v.len()); 
    for (i, &x) in v.iter().enumerate() {
        r += i - t.sum(x) as usize;
        t.add(x, 1);
    }
    r
}

}