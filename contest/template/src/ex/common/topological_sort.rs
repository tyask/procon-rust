#![allow(dead_code)]
use crate::common::*;

pub fn topological_sort(g: &Graph) -> Option<Vec<us>> {
    let n = g.len();

    let mut deg = vec![0; n];
    for v in 0..n { for &u in &g[v] { deg[u] += 1; }}

    let mut que = deque::new();
    for v in 0..n { if deg[v] == 0 { que.push_back(v); }}

    let mut ret = Vec::with_capacity(n);
    while let Some(v) = que.pop_front() {
        ret.push(v);
        for &u in &g[v] { deg[u]-=1; if deg[u]==0 { que.push_back(u); }}
    }

    if ret.len() == n { Some(ret) } else { None }
}