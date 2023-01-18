#![allow(dead_code)]
use std::collections::VecDeque;

use crate::common::*;

fn topological_sort(g: &Graph) -> Vec<us> {
    let n = g.len();

    let mut deg = vec![0; n];
    for v in 0..n { for &u in &g[v] { deg[u] += 1; }}

    let mut que = VecDeque::new();
    for v in 0..n { if deg[v] == 0 { que.push_back(v); }}

    let mut ret = Vec::with_capacity(n);
    while let Some(v) = que.pop_front() {
        ret.push(v);
        for &u in &g[v] { deg[u]-=1; if deg[u]==0 { que.push_back(u); }}
    }

    ret
}