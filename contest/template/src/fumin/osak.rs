#![allow(dead_code)]
use crate::{common::*, chmin};

// 高速素因数分解
pub struct Osak {
    minfactor: Vec<us>,
}

impl Osak {
    pub fn new(n: us) -> Self {
        // minfactor[i] := iの最小の素因数
        // エラストテネスの篩みたいな感じで求める. O(NloglogN)
        let mut minfactor = (0..=n).cv();
        let mut i = 2;
        while i*i <= n {
            for j in (i..=n).step_by(i) { chmin!(minfactor[j], i); }
            i += 1;
        }
        Self { minfactor: minfactor }
    }

    pub fn prime_fact(&self, mut k: us) -> bmap<us, us> {
        assert!(k < self.minfactor.len());
        // 最小の素因数で割っていくことで素因数分解する. O(logN)
        let mf = &self.minfactor;
        let mut m = bmap::new();
        while k > 1 {
            *m.or_def_mut(&mf[k]) += 1;
            k /= mf[k];
        }
        m
    }
}
