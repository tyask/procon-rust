#![allow(dead_code)]
use crate::common::*;

pub struct WeightedUnionfind {
    d: Vec<i64>,
    weight: Vec<i64>,
}

impl WeightedUnionfind {
    pub fn new(n:us) -> Self { Self{ d: vec![-1; n] , weight: vec![0; n] } }

    pub fn build(n:us, xyw: &[(us,us,i64)]) -> Self {
        let mut uf = Self::new(n);
        for &(x,y,w) in xyw { uf.unite(x, y, w); }
        uf
    }

    pub fn root(&mut self, x: us) -> us {
        if self.d[x] < 0 {
            x
        } else {
            let r = self.root(self.d[x].us()).i64();
            self.weight[x] += self.weight[self.d[x].us()];
            self.d[x] = r;
            self.d[x].us()
        }
    }

    pub fn unite(&mut self, x: us, y: us, w: i64) -> bool {
        let (mut rx, mut ry) = (self.root(x), self.root(y));
        if rx == ry { return self.diff_weight(x, y).is_some_and(|d|d==w); }
        let mut w = w + self.weight(x) - self.weight(y);

        if self.d[rx] > self.d[ry] { std::mem::swap(&mut rx, &mut ry); w = -w; }
        self.d[rx] += self.d[ry];
        self.d[ry] = rx.i64();
        self.weight[ry] = w;
        true
    }
    pub fn same(&mut self, x: us, y: us) -> bool { self.root(x) == self.root(y) }
    pub fn size(&mut self, x: us) -> us { let p = self.root(x); -self.d[p] as us }

    pub fn weight(&mut self, x: us) -> i64 { self.root(x); self.weight[x] }
    pub fn diff_weight(&mut self, x: us, y: us) -> Option<i64> {
        if self.same(x, y) { Some(self.weight(y) - self.weight(x)) } else { None }
    }
}

impl WeightedUnionfind {
    pub fn groups(&mut self) -> Vec<Vec<us>> {
        let n = self.d.len();
        let (mut root_buf, mut group_size, mut res) = (vec![0; n], vec![0; n], vec![vec![]; n]);
        for i in 0..n { root_buf[i] = self.root(i); group_size[root_buf[i]]+=1; }
        for i in 0..n { res[i].reserve(group_size[i]); }
        for i in 0..n { res[root_buf[i]].push(i); }
        res.iter().filter(|v|v.len()>0).cloned().cv()
    }
}

