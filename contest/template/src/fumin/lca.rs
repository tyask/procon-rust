#![allow(dead_code)]
use std::{*, mem::swap};
use crate::common::*;
use super::graph::Graph;

// CAP(fumin::graph)

// Lowest Common Ancestor 
pub struct Lca {
    pub dist: Vec<us>, // rootからの距離
    parent: Vec<Vec<us>>, // parent[k][u]:=uの2^k先の親
}

impl Lca {
    const NIL: us = us::INF;
    pub fn new(g: &Graph, root: us) -> Self { 
        let n = g.len();
        let k = Self::largest_pow2(n);
        let mut parent = vec![vec![Self::NIL; n]; k];
        let mut dist = vec![Self::NIL; n];

        let mut st = deque::new();
        st.push_back((root, Self::NIL, 0));
        while let Some((v, p, d)) = st.pop_back() {
            parent[0][v] = p; // 2^0=1先の親(つまり直近の親)
            dist[v] = d; // rootからの距離
            for &nv in &g[v] { if nv != p { st.push_back((nv, v, d+1)); } }
        }

        // ダブリングの要領で各点の2^k上先の親を計算しておく.
        for ki in 0..k-1 { for v in 0..n {
            let p = parent[ki][v];
            parent[ki+1][v] = if p != Self::NIL { parent[ki][p] } else { Self::NIL };
        }}

        Self { parent: parent, dist: dist }
    }

    pub fn lca(&self, mut u: us, mut v: us) -> us {
        let (parent, dist) = (&self.parent, &self.dist);

        if dist[u] < dist[v] { swap(&mut u, &mut v); } // uを深くする
        let k = parent.len();

        // uのLCAまでの距離をvに揃える.
        // ダブリングの要領で距離の差を2のべき乗ごとに詰めていくイメージ
        for ki in 0..k {
            if (dist[u]-dist[v])>>ki & 1 == 1 { u = self.parent[ki][u]; }
        }

        // u/vからrootのどこかにLCAはあるはずなので二分探索っぽくrootまで遡っていくイメージでLCAを探す.
        // kが十分に大きくLCAがp[k][u]/p[k][v]より小さい場合、p[k][u]==p[k][v]となる(LCAの地点で合流するため). この場合は何もせず次に小さいkを調べる.
        // kが小さく、LCAがp[k][u]/p[k][v]より大きい場合、p[k][u]!=p[k][v]となるため、u/vをその地点まで移動する.
        // u/vを移動させた後のu/vとLCAの距離はk以下であるため、その後は引き続き次に小さいkを見ていけばOK.
        // 最終的にu/vはLCAの一つ手前まで来ているので、1つ親がLCAとなる.
        if u == v { return u; }
        for ki in (0..k).rev() {
            if parent[ki][u] != parent[ki][v] {
                u = parent[ki][u];
                v = parent[ki][v];
            }
        }
        parent[0][u]
    }

    // u, v間の距離
    pub fn dist(&self, u: us, v: us) -> us {
        self.dist[u] + self.dist[v] - 2 * self.dist[self.lca(u, v)]
    }

    fn largest_pow2(n: usize) -> usize {
        let mut k=1; while (1<<k) < n { k+=1; } k
    }
}
