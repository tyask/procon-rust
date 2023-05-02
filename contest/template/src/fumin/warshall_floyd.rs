#![allow(dead_code)]
use itertools::iproduct;
use crate::{common::*, chmin};

pub fn warshall_floyd(dist: &Vec<Vec<is>>) -> Vec<Vec<is>> {
    let mut d = dist.clone();
    let n = dist.len();
    for (k, i, j) in iproduct!(0..n, 0..n, 0..n) {
        chmin!(d[i][j], d[i][k] + d[k][j]);
    }
    d
}