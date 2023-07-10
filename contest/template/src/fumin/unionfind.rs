#![allow(dead_code)]
use crate::common::*;

pub struct Unionfind {
    d: Vec<i64>
}

impl Unionfind {
    pub fn new(n:us) -> Self { Self{ d: vec![-1; n] } }

    pub fn build(n:us, xy: &[(us,us)]) -> Self {
        let mut uf = Self::new(n);
        for &(x,y) in xy { uf.unite(x, y); }
        uf
    }

    pub fn leader(&mut self, x: us) -> us {
        if self.d[x] < 0 {
            x
        } else {
            self.d[x] = self.leader(self.d[x] as us) as i64;
            self.d[x] as us
        }
    }

    pub fn unite(&mut self, x: usize, y: usize) -> bool {
        let (mut x, mut y) = (self.leader(x), self.leader(y));
        if x == y { return false }
        if self.d[x] > self.d[y] { std::mem::swap(&mut x, &mut y) }
        self.d[x] += self.d[y];
        self.d[y] = x as i64;
        true
    }
    pub fn same(&mut self, x: us, y: us) -> bool { self.leader(x) == self.leader(y) }
    pub fn size(&mut self, x: us) -> us { let p = self.leader(x); -self.d[p] as us }
}

impl Unionfind {
    pub fn groups(&mut self) -> Vec<Vec<us>> {
        let n = self.d.len();
        let (mut leader_buf, mut group_size, mut res) = (vec![0; n], vec![0; n], vec![vec![]; n]);
        for i in 0..n { leader_buf[i] = self.leader(i); group_size[leader_buf[i]]+=1; }
        for i in 0..n { res[i].reserve(group_size[i]); }
        for i in 0..n { res[leader_buf[i]].push(i); }
        res.iter().filter(|v|v.len()>0).cloned().cv()
    }
}

