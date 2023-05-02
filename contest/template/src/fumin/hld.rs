#![allow(dead_code)]
use crate::common::*;
use super::graph::Graph;

// CAP(fumin::graph)

pub struct HLD {
    g: Graph,
    roots: Vec<us>,
    parent: Vec<us>,
    left: Vec<us>,
    right: Vec<us>,
    vertex: Vec<us>,
}
 
impl HLD {
    pub fn new(g: &Graph, root: us) -> Self {
        assert!(g.len() <= 10usize.pow(8));
        let mut g = g.clone();
        let n = g.len();

        let (parent, path) = Self::build_parent(&mut g, root);
        Self::build_heavy_edges(&mut g, &path);

        let mut roots = (0..n).cv(); // roots[i]=heavy edgeによる連結成分におけるiのルートノード
        let mut left = vec![0; n];
        let mut right = vec![0; n];
        let mut dfs = vec![(root, false)];
        let mut id = 0;
        while let Some((v, end)) = dfs.pop() {
            if end { right[v] = id; continue; }
            left[v] = id;
            id += 1;
            dfs.push((v, true));
            let child = &g[v];
            if !child.is_empty() {
                // light edge
                for &u in child[1..].iter() {
                    roots[u] = u;
                    dfs.push((u, false));
                }

                // heavy edge
                let u = child[0];
                roots[u] = roots[v];
                dfs.push((u, false));
            }
        }

        let mut vertex = vec![n; n];
        for (i, &l) in left.iter().enumerate() { vertex[l] = i; }

        HLD {
            g: g,
            roots: roots,
            parent: parent,
            left: left,
            right: right,
            vertex: vertex,
        }
    }

    fn build_parent(g: &mut Graph, root: us) -> (Vec<us>, Vec<us>) {
        // parent[i] = iの親
        // path = rootから辿った順番 (多分浅い順)
        let n = g.len();
        let mut parent = vec![n; n];
        let mut path = Vec::with_capacity(n);
        path.push(root);
        parent[root] = root;
        for i in 0..n {
            let v = path[i];
            for u in g[v].clone() {
                assert!(parent[u] == n);
                parent[u] = v;
                g[u].retain(|&e| e != v); // 親を削除
                path.push(u);
            }
        }
        (parent, path)
    }

    fn build_heavy_edges(g: &mut Graph, path: &Vec<us>) {
        // sum[i]=iを頂点とする部分木のサイズ
        // g[i][0]にheavy edgeを置く.
        let mut sum = vec![1; g.len()];
        for &v in path.iter().rev() { // 深い頂点から順に部分木のサイズを計算していく.
            let child = &mut g[v];
            if !child.is_empty() {
                let (pos, _) = child.iter().enumerate().max_by_key(|p| sum[*p.1]).unwrap();
                child.swap(0, pos);
                sum[v] = 1 + child.iter().fold(0, |s, a| s + sum[*a]);
            }
        }
    }

    // aとbのLCA
    pub fn lca(&self, mut a: us, mut b: us) -> us {
        assert!(a < self.g.len() && b < self.g.len());
        let (roots, parent, left) = (&self.roots, &self.parent, &self.left);
        while roots[a] != roots[b] {
            if left[a] > left[b] { std::mem::swap(&mut a, &mut b); } // bを深いノードとする.
            b = parent[roots[b]]; // bを1つ上の連結成分に移動する.
        }
        std::cmp::min((left[a], a), (left[b], b)).1
    }

    pub fn path0(&self, s: us, t: us) -> Vec<(us,us)> {
        let (up, down) = self.path(s, t);
        up.into_iter().chain(down).cv()
    }

    // s -> t のパスの各連結成分における区間(半開区間)
    pub fn path(&self, s: us, t: us) -> (Vec<(us,us)>, Vec<(us, us)>) {
        assert!(s < self.g.len() && t < self.g.len());
        let mut up = vec![];
        let mut down = vec![];
        let (roots, parent, left) = (&self.roots, &self.parent, &self.left);
        let mut x = s;
        let mut y = t;
        while roots[x] != roots[y] {
            if left[x] > left[y] {
                let p = roots[x];
                up.push((left[p], left[x] + 1));
                x = parent[p];
            } else {
                let p = roots[y];
                down.push((left[p], left[y] + 1));
                y = parent[p];
            }
        }
        if left[x] > left[y] {
            up.push((left[y] + 1, left[x] + 1));
        } else {
            down.push((left[x] + 1, left[y] + 1));
        }
        down.reverse();

        (up, down)
    }

    pub fn sub_tree(&self, v: us) -> (us, us) {
        assert!(v < self.g.len());
        (self.left[v], self.right[v])
    }

    pub fn index(&self, v: us) -> us { self.left[v] }
    pub fn vertex(&self, i: us) -> us { self.vertex[i] }

    pub fn parent(&self, v: us) -> Option<us> {
        assert!(v < self.g.len());
        let p = self.parent[v];
        if p == v { None } else { Some(p) }
    }

    // s -> t へのパスの2番目の頂点を返す
    pub fn next(&self, s: us, t: us) -> us {
        assert!(s < self.g.len() && t < self.g.len() && s != t);
        let (a, b) = self.sub_tree(s);
        let (c, d) = self.sub_tree(t);
        if !(a <= c && d <= b) { // tがsの部分木に含まれない場合
            return self.parent[s];
        }

        // tがsの部分木の場合、tをsの連結成分まで移動させる.
        // ループを抜けた時点で、sとposは同じ連結成分内にある.
        // s!=posの場合、sのheavy edge先のノードが2番目の頂点となる.
        // s==posの場合、移動前の頂点(pre)が2番目の頂点となる.
        let (mut pos, mut pre) = (t, t);
        while self.roots[s] != self.roots[pos] {
            pre = self.roots[pos];
            pos = self.parent[pre];
        }
        if s != pos { self.g[s][0] } else { pre }
    }

    pub fn jump(&self, s: us, t: us, mut k: us) -> Option<us> {
        assert!(s.max(t) < self.g.len());
        let (mut up, mut down) = self.path(s, t);
        for (l, r) in up.drain(..) {
            if k < r - l { return Some(self.vertex[r - 1 - k]); }
            k -= r - l;
        }
        for (l, r) in down.drain(..) {
            if k < r - l { return Some(self.vertex[l + k]); }
            k -= r - l;
        }
        None
    }
}
