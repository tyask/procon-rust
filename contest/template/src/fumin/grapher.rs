#![allow(dead_code)]
use crate::common::*;
use super::graph::Graph;

// CAP(fumin::graph)

impl Graph {
    // 最短距離を2度計算することで木の直径を求める. (u:Node1, v:Node2, d:Distance)
    pub fn diameter(&self) -> (us, us, us) {
        let u = self.bfs(0).iter().enumerate().max_by_key(|p|p.1).map(|p|p.0).unwrap();
        let (v, &d) = self.bfs(u).iter().enumerate().max_by_key(|p|p.1).unwrap();
        (u, v, d)
    }
}

impl Graph {
    // 2部グラフの色を0,1で塗り分ける. 2部グラフでない場合Err.
    // g: グラフを表す隣接リスト
    pub fn colorize_bipartite(&self) -> Result<Vec<is>,()> {
        let mut col = vec![-1; self.len()];
        for i in 0..self.len() {
            if col[i] >= 0 { continue; }
            let mut st = deque::new();
            st.push_back((i, 0));
            while let Some((v, c)) = st.pop_back() {
                if col[v] == c { return Err(()); }
                col[v] = c;
                for &u in &self[v] { if col[u] < 0 { st.push_back((u, 1-c)); } }
            }
        }
        Ok(col)
    }
}

impl Graph {
    // 強連結成分分解(scc)
    pub fn scc(&self) -> Vec<Vec<us>> {
        let n = self.len();

        // 帰りがけ順に点を記録する->t
        let mut t = vec![];
        let mut vis = vec![false; n];
        let mut st = deque::new();
        for i in 0..n {
            st.push_back((i, true));
            while let Some((v, pre)) = st.pop_back() {
                if pre {
                    if vis[v] { continue; }          
                    vis[v] = true;
                    st.push_back((v, false));
                    for &u in &self[v] { st.push_back((u, true)); }
                } else {
                    t.push(v);
                }
            }
        }

        // tの各点から逆向きにdfs
        // 各点から到達できる点を強連結成分とする.
        let h = self.rev();
        let mut g = vec![us::INF; n];
        let mut gi = 0;
        let mut st = deque::new();
        for &i in t.iter().rev() {
            if g[i] != us::INF { continue; }
            st.push_back(i);
            while let Some(v) = st.pop_back() {
                if g[v] != us::INF { continue; }
                g[v] = gi;
                for &u in &h[v] { st.push_back(u); }
            }
            gi += 1;
        }

        let mut grp = vec![vec![]; gi];
        for i in 0..n { grp[g[i]].push(i); }

        grp
        
    }
}

impl Graph {
    pub fn topological_sort(&self) -> Option<Vec<us>> {
        let n = self.len();

        let mut deg = vec![0; n];
        for v in 0..n { for &u in &self[v] { deg[u] += 1; }}

        let mut que = deque::new();
        for v in 0..n { if deg[v] == 0 { que.push_back(v); }}

        let mut ret = Vec::with_capacity(n);
        while let Some(v) = que.pop_front() {
            ret.push(v);
            for &u in &self[v] { deg[u]-=1; if deg[u]==0 { que.push_back(u); }}
        }

        if ret.len() == n { Some(ret) } else { None }
    }
}

impl Graph {
    // DAGの最長経路を求める. O(N)
    pub fn longest_dist(&self) -> us {
        let n = self.len();
        let mut dp = vec![0; n]; // dp[i]:=iからの最長経路
        let mut vis = vec![false; n];
        for v in 0..n {
            if vis[v] { continue; }
            let mut st = deque::new();
            st.push_back((v,true));
            while let Some((v,pre)) = st.pop_back() {
                if pre {
                    if vis[v] {
                        if dp[v]==0 { return us::INF; } else { continue; }
                    }
                    vis[v] = true;
                    st.push_back((v,false));
                    for &u in &self[v] { st.push_back((u,true)); }
                } else {
                    dp[v] = self[v].iter().map(|&u|dp[u]).max().unwrap_or_default() + 1;
                }
            }
        }
        return dp.vmax();
    }
}