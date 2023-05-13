#![allow(dead_code)]
use std::*;
use crate::common::*;
use super::tree::Tree;

// CAP(fumin::graph)

pub struct EulerTour {
    pub vin:  Vec<us>, // 頂点iを最初に通った時のpathのindex
    pub vout: Vec<us>, // 頂点iを最後に通った時のpathのindex
    pub path: Vec<us>, // DFSで通った頂点の列
}

impl EulerTour {
    pub fn new(tree: &Tree, s: us) -> Self {
        let mut st = deque::new();
        st.push_back((s, s, true));
        let (mut vin, mut vout, mut path) = (vec![0; tree.len()], vec![0; tree.len()], vec![]);
        while let Some((v, p, pre)) = st.pop_back() {
            if pre {
                vin[v] = path.len();
                let mut is_leaf = true;
                for &u in &tree[v] { if u != p {
                    is_leaf = false;
                    st.push_back((v, p, false));
                    st.push_back((u, v, true));
                }}
                if is_leaf { vout[v] = path.len(); }
            } else {
                vout[v] = path.len();
            }
            path.push(v);
        }

        Self { vin: vin, vout: vout, path: path }
    }
}
