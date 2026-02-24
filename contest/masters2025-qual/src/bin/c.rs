#![allow(unused_imports)]
use std::{*, collections::*, ops::*, cmp::*, iter::*};
use grid_v::GridV;
use proconio::{input, fastout};
use common::*;
use fumin::*;
use pt::Dir;
use itertools::*;
use rand::{seq::IteratorRandom as _, SeedableRng};
use time::Instant;

fn main() {
    solve();
}

type P = pt::Pt<us>;
const N: us = 20;

struct Io {
    _n:us,
    m:us,
    g: GridV<char>,
    a: P,
    b: P,
    c: P,
}

impl Io {
    fn new() -> Self {
        input! {n:us,m:us,cx:[chars;n]}
        let mut g = GridV::new(n,n);
        let mut a = P::INF;
        let mut b = P::INF;
        let mut c = P::INF;
        for i in 0..n { for j in 0..n {
            g[i][j] = cx[i][j];
            if g[i][j]=='A' { a = P::new(i,j); }
            if g[i][j]=='B' { b = P::new(i,j); }
            if g[i][j]=='C' { c = P::new(i,j); }
        }}
        Self {
            _n: n,
            m,
            g,
            a,
            b,
            c,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ActionType {
    Move = 1,
    Carry = 2,
    Roll = 3,
}

#[derive(Debug, Clone)]
struct Action {
    t: ActionType,
    d: Dir,
}

impl ToString for Action {
    fn to_string(&self) -> String {
        format!("{} {}", self.t as us, self.d.c())
    }
}

struct Out {
    actions: Vec<Action>,
}

impl Out {
    fn new() -> Self {
        Self {
            actions: vec![],
        }
    }

    #[fastout]
    fn out(&self) {
        for a in &self.actions {
            println!("{}", a.to_string());
        }

    }
}

struct Solver<'a> {
    io: &'a Io,
    g: GridV<char>,
    p: P,
    fb: fast_bit_set::FastBitSet2d,
}

impl<'a> Solver<'a> {
    fn new(io: &'a Io) -> Self {
        Self {
            io,
            g: io.g.clone(),
            p: io.a,
            fb: fast_bit_set::FastBitSet2d::new(N,N),
        }
    }

    fn solve(&mut self, rng: &mut impl rand_core::RngCore) -> Out {
        // 大まかな方針
        // 1) 穴から見て上下左右の穴又は鉱石を穴に落とす
        // 2) 何かしらの鉱石を探し、それを穴に落とす
        //    ※穴に落とす場合は穴と一直線上にあるマスに到達したらそこからは転がすようにする
        // 3) 岩を壊さない範囲内に鉱石が見つからなかったら適当な鉱石に到達できるように岩を壊し、2)に戻る

        let io = self.io;
        let mut out = Out::new();
        // 穴から見て上下左右の直線上にある岩を穴に落とす
        let (clears, locks) = self.clear_locks(io.a);
        out.actions.extend(clears);
        locks.iter().for_each(|&v|self.g[v] = '.');

        while out.actions.len() < 10000 {
            // 何かしらの鉱石へのパスの候補を探す (ただし岩を越えない範囲で)
            let ps_cand = self.find_any_ores(self.p, &['@'], &rustc_hash::FxHashSet::default());
            if ps_cand.is_empty() {
                // 鉱石が見つからない場合、鉱石がある場所に行けるように岩を壊す
                if !self.clear_locks_for_unreached(&mut out, rng) {
                    // 壊す岩がない場合、鉱石を落とし終えたということで処理終了
                    break;
                } else {
                    continue;
                }
            }

            // 実際に回収する鉱石へのパスを選ぶ
            // いい感じのパスを選びたいが、ここでは適当に最短経路のパスを選択している
            let ps = ps_cand.iter().sorted_by_key(|&p|p.len()).take(1).choose(rng).unwrap();
            let t = ps.last().cloned().unwrap();
            let t0 = P::new(self.p.x, t.y);
            // 鉱石の場所までは岩や穴をよける必要がないので直線に最短路を進む
            out.actions.extend(vec![Action {t:ActionType::Move, d: self.to_dir(self.p, t0)}; self.p.manhattan_distance(t0)]);
            out.actions.extend(vec![Action {t:ActionType::Move, d: self.to_dir(t0, t)}; t.manhattan_distance(t0)]);

            // 鉱石の場所を空白にする
            self.g[ps.last().cloned().unwrap()] = '.';

            // 自分の位置を更新
            self.p = t;

            // 鉱石から穴又は穴から上下左右の直線上のマスまでの経路を計算
            let rev = self.find(&['A'], &self.lookup_reached_by_roll(io.a), &['@', 'a']);
            let mut np = io.a;
            for (&u,&v) in rev.iter().tuple_windows() {
                if let Some(d) = self.can_reach_by_roll(u, io.a) {
                    // もし上下左右の直線上のマスに到達したら穴に転がす
                    out.actions.push(Action { t: ActionType::Roll, d});
                    np = u;
                    break;
                } else {
                    // そうでない場合普通に運ぶ
                    out.actions.push(Action { t: ActionType::Carry, d: Dir::dir(u, v)});
                }
            }

            if let Some(v) = rev.last().cloned() {
                if v != io.a {
                    // revの最後のマスが穴出ない場合はここから穴に向かって転がす
                    // (この処理は本当は上のfor文の中で一緒にできたほうが良いが、うまくいかないので適当にここでやっている)
                    out.actions.push(Action { t: ActionType::Roll, d: self.to_dir(v, io.a)});
                    np = v;
                }
            }
            self.p = np;
            // eprintln!("{}", self.g.to_string());
        }

        out.actions.truncate(10000);
        out
    }

    // sをスタート地点として、a/b/cまでの経路を探索する
    // ただし、exに含まれる文字、またはexpに含まれる点は通らないようにする
    fn find_any_ores(&mut self, s: P, ex: &[char], exp: &rustc_hash::FxHashSet<P>) -> Vec<Vec<P>> {
        let mut q = deque::new();
        q.push((s, P::INF));
        self.fb.clear(); // 既に通ったマスは2度通らないようにマークする
        let mut prev = rustc_hash::FxHashMap::default();
        let mut ret = vec![];
        while let Some((v,p)) = q.pop_front() {
            if self.fb[v] { continue; }
            self.fb.set(v,true);
            prev.insert(v,p);
            if ['a', 'b', 'c'].contains(&self.g[v]) { // a/b/cに到達した
                // a/b/cへの経路を復元する
                let mut ps = vec![];
                let mut t = v;
                ps.push(t);
                while prev[&t] != P::INF {
                    t = prev[&t];
                    ps.push(t);
                }
                ps.reverse();
                ret.push(ps);
            }

            // 今いる地点から上下左右に移動する
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if !self.g.is_in_p(nv) || ex.contains(&self.g[nv]) || exp.contains(&nv) { continue; }
                q.push((nv, v));
            }
        }
        ret
    }


    // fn clear_locks_for_unreached2(&mut self, out: &mut Out, rng: &mut impl rand_core::RngCore) -> Vec<P> {
    //     let mut q = deque::new();
    //     q.push((self.io.a, P::INF));
    //     self.fb.clear();
    //     let mut prev = rustc_hash::FxHashSet::default();
    //     let mut ps = vec![];
    //     while let Some((v,p)) = q.pop_front() {
    //         if self.fb[v] { continue; }
    //         self.fb.set(v,true);
    //         prev.insert(v,p);
    //         if self.g[v] == 'a' {
    //             let mut t = v;
    //             ps.push(t);
    //             while prev[&t] != P::INF {
    //                 t = prev[&t];
    //                 ps.push(t);
    //             }
    //             ps.reverse();
    //             return ps;
    //         }

    //         for d in Dir::VAL4 {
    //             let nv = v.next(d);
    //             if !self.g.is_in_p(nv) { continue; }
    //             if self.g[nv] == '.' { q.push_front((nv, v)); } else { q.push((nv,v)); }
    //         }
    //     }
    //     ps
    // }

    fn clear_locks_for_unreached(&mut self, out: &mut Out, rng: &mut impl rand_core::RngCore) -> bool {
        // 盤面から鉱石の場所を探す
        // どの鉱石を選ぶかは現在地から近い場所から適当に決める (最適化できる?)
        let op = P::iter(0..N, 0..N)
            .filter(|&v|self.g[v].is_lowercase())
            .sorted_by_key(|&v|v.manhattan_distance(self.io.a))
            .take(3)    
            .choose(rng);
            // .min_by_key(|&v|v.manhattan_distance(self.io.a));
            // .min_by_key(|&v|v.manhattan_distance(self.p));
        
        // 見つからない場合終了
        let Some(t) = op else { return false; };
        // let ps = self.clear_locks_for_unreached2(out, rng);
        // if ps.is_empty() { return false;}
        // let t = ps.last().cloned().unwrap();

        // 簡単のため一旦穴に戻る
        let ps = self.find(&vec![self.g[t].to_ascii_uppercase()], &rustc_hash::FxHashSet::default(), &['@']);
        for (&u,&v) in ps.iter().tuple_windows() {
            out.actions.push(Action { t: ActionType::Move, d: Dir::dir(u, v)});
        }

        // 穴の位置から目的の鉱石の位置まで岩を除去しながら進む
        // 横=>縦の順に進むか、縦=>横の順に進むか、効率の良い方を採用する
        // 一度曲がるまでは岩は直接穴に転がすが、一度曲がった場合は角まで運んでから穴に転がす
        let mut v0 = ps.last().cloned().unwrap();
        let mut v1 = ps.last().cloned().unwrap();
        let mut a0 = vec![];
        let mut a1 = vec![];
        let mut g0 = self.g.clone();
        let mut g1 = self.g.clone();
        {
            let mut v = v0; 
            let ty = P::new(v.x, t.y);
            let dy = self.to_dir(v, ty);
            for _ in 0..v.y.abs_diff(t.y) {
                v = v.next(dy);
                a0.push(Action {t: ActionType::Move, d: dy});
                if g0[v] == '@' || g0[v].is_lowercase() {
                    a0.push(Action {t: ActionType::Roll, d: dy.rev()});
                    g0[v] = '.';
                }
            }
    
            let dx = self.to_dir(ty, t);
            for _ in 0..ty.x.abs_diff(t.x) {
                v = v.next(dx);
                a0.push(Action {t: ActionType::Move, d: dx});
                if g0[v] == '@' || g0[v].is_lowercase() {
                    a0.extend(vec![Action {t: ActionType::Carry, d: dx.rev()}; v.manhattan_distance(ty)]);
                    a0.push(Action {t: ActionType::Roll, d: dy.rev()});
                    a0.extend(vec![Action {t: ActionType::Move, d: dx}; v.manhattan_distance(ty)]);
                    g0[v] = '.';
                }
            }
            v0 = v;
        }

        {
            let mut v = v1;
            let tx = P::new(t.x, v.y);
            let dx = self.to_dir(v, tx);
            for _ in 0..v.x.abs_diff(t.x) {
                v = v.next(dx);
                a1.push(Action {t: ActionType::Move, d: dx});
                if g1[v] == '@' || g1[v].is_lowercase() {
                    a1.push(Action {t: ActionType::Roll, d: dx.rev()});
                    g1[v] = '.';
                }
            }
    
            let dy = self.to_dir(tx, t);
            for _ in 0..tx.y.abs_diff(t.y) {
                v = v.next(dy);
                a1.push(Action {t: ActionType::Move, d: dy});
                if g1[v] == '@' || g1[v].is_lowercase() {
                    a1.extend(vec![Action {t: ActionType::Carry, d: dy.rev()}; v.manhattan_distance(tx)]);
                    a1.push(Action {t: ActionType::Roll, d: dx.rev()});
                    a1.extend(vec![Action {t: ActionType::Move, d: dy}; v.manhattan_distance(tx)]);
                    g1[v] = '.';
                }
            }
            v1 = v;
        }
        // debug!(a0);
        // debug!(a1);
        if a0.len() <= a1.len() {
            out.actions.extend(a0);
            self.g = g0;
            self.p = v0;
            // out.actions.extend(a1);
            // self.g = g1;
            // self.p = v1;
        } else {
            out.actions.extend(a1);
            self.g = g1;
            self.p = v1;
        }
        true
    }

    fn clear_locks(&self, x:P) -> (Vec<Action>, Vec<P>) {
        let mut res = vec![];
        let mut pos = vec![];
        for d in Dir::VAL4 { // 4方向
            let mut v = x;
            let mut aa = vec![];
            let mut pv = v;
            for _ in 0..N {
                v = v.next(d);
                if !self.g.is_in_p(v) { break; }
                // 岩か鉱石が見つかったら穴に転がす
                if self.g[v] == '@' || self.g[v].is_lowercase() {
                    aa.extend(vec![Action{t: ActionType::Move, d}; v.manhattan_distance(pv)]);
                    aa.push(Action{t: ActionType::Roll, d: d.rev()});
                    pv = v;
                    pos.push(v);
                }
            }

            // 穴まで戻る
            if !aa.is_empty() {
                res.extend(aa);
                res.extend(vec![Action{t:ActionType::Move, d: d.rev()}; pv.manhattan_distance(x)]);
            }
        }
        (res, pos)
    }

    fn to_dir(&self, v:P, t:P) -> Dir {
        if v.x == t.x {
            if v.y < t.y { Dir::R } else { Dir::L }
        } else {
            if v.x < t.x { Dir::D } else { Dir::U }
        }
    }

    fn can_reach_by_roll(&self, v: P, t: P) -> Option<Dir> {
        if v == t { return None; }
        let d = self.to_dir(v, t);

        let mut v = v;
        for _ in 0..N {
            v = v.next(d);
            if !self.g.is_in_p(v) || self.g[v]=='@' || self.g[v].is_lowercase() { break; }
            if v == t { return Some(d); }
        }
        None
    }

    fn lookup_reached_by_roll(&self, v:P) -> rustc_hash::FxHashSet<P> {
        let mut s = rustc_hash::FxHashSet::default();
        for d in Dir::VAL4 {
            let mut v = v;
            for _ in 0..N {
                v = v.next(d);
                if !self.g.is_in_p(v) || self.g[v]=='@' || self.g[v].is_lowercase() { break; }
                s.insert(v);
            }
        }
        s
    }

    // tに含まれる文字のマス、又はtpに含まれるマスへの経路を探索する
    // ただし、exに含まれるマスは通らない
    fn find(&mut self, t: &[char], tp: &rustc_hash::FxHashSet<P>, ex: &[char]) -> Vec<P> {
        let mut q = deque::new();
        q.push((self.p, P::INF));
        self.fb.clear();
        let mut prev = rustc_hash::FxHashMap::default();
        let mut ps = vec![];
        while let Some((v,p)) = q.pop_front() {
            if self.fb[v] { continue; }
            self.fb.set(v,true);
            prev.insert(v,p);
            if t.contains(&self.g[v]) || tp.contains(&v) { // 目的地に到達
                // 経路復元
                let mut t = v;
                ps.push(t);
                while prev[&t] != P::INF {
                    t = prev[&t];
                    ps.push(t);
                }
                ps.reverse();
                return ps;
            }

            // 隣のマスに進む
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if !self.g.is_in_p(nv) || ex.contains(&self.g[nv]) { continue; }
                q.push((nv, v));
            }
        }
        ps
    }
}

// CONTEST(abcXXX-a)
// #[fastout]
fn solve() {
    let st = Instant::now();
    let io = Io::new();
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let mut out = Out::new();
    let mut mn = us::INF;
    while st.elapsed().as_millis() < 1950 {
        let mut solver = Solver::new(&io);
        let o = solver.solve(&mut rng);
        if chmin!(mn, o.actions.len()) {
            out = o;
        }
        break;
    }
    out.out();
}

pub mod beam {
use crate::*;

#[allow(non_camel_case_types)]
type uint=u16;

struct Input {
    n: us,
    m: us,
    g: GridV<char>,
    a: P,
    b: P,
    c: P,
}

impl Input {
    fn new() -> Self {
        input! {n:us,m:us,cx:[chars;n]}
        let mut g = GridV::new(n,n);
        let mut a = P::INF;
        let mut b = P::INF;
        let mut c = P::INF;
        for i in 0..n { for j in 0..n {
            g[i][j] = cx[i][j];
            if g[i][j]=='A' { a = P::new(i,j); }
            if g[i][j]=='B' { b = P::new(i,j); }
            if g[i][j]=='C' { c = P::new(i,j); }
        }}
        Self {
            n,
            m,
            g,
            a,
            b,
            c,
        }
    }
}

#[derive(Clone,PartialEq)]
struct State{}
impl State{
    fn new(input:&Input)->State{
        todo!();
    }

    fn apply(&mut self,node:&Node){
        todo!();
    }

    fn revert(&mut self,node:&Node){
        todo!();
    }
}


#[derive(Clone)]
struct Cand{
    op:u8,
    parent:uint,
    eval_score:i64,
    hash:u64,
}
impl Cand{
    fn raw_score(&self,input:&Input)->i64{
        todo!();
    }
    
    fn to_node(&self)->Node{
        Node{
            child:!0,
            prev:!0,
            next:!0,
            op:self.op,
            parent:self.parent,
        }
    }
}


#[derive(Clone,Default)]
struct Node{
    op:u8,
    parent:uint,
    child:uint,
    prev:uint,
    next:uint,
}


const MAX_WIDTH:usize=1000;
const TURN:usize=100;
const MAX_NODES:usize=MAX_WIDTH*5;


struct BeamSearch{
    state:State,
    leaf:Vec<uint>,
    next_leaf:Vec<uint>,
    nodes:Vec<Node>,
    cur_node:usize,
    free:Vec<uint>,
}
impl BeamSearch{
    fn new(state:State,node:Node)->BeamSearch{
        assert!(MAX_NODES<uint::MAX as usize,"uintのサイズが足りないよ");

        let mut nodes=vec![Node::default();MAX_NODES];
        nodes[0]=node;

        let mut leaf=Vec::with_capacity(MAX_WIDTH);
        leaf.push(0);
        let next_leaf=Vec::with_capacity(MAX_WIDTH);
        let free=(1..nodes.len() as uint).rev().collect();

        BeamSearch{
            state,nodes,free,
            leaf,next_leaf,
            cur_node:0,
        }
    }

    fn reset(&mut self,state:State,node:Node){
        self.state=state;
        self.nodes[0]=node;
        self.leaf.clear();
        self.leaf.push(0);
        self.next_leaf.clear();
        self.free.clear();
        self.free.extend((1..self.nodes.len() as uint).rev());
        self.cur_node=0;
    }
    
    fn add_node(&mut self,cand:Cand){
        let next=self.nodes[cand.parent as usize].child;
        
        let new=if let Some(n)=self.free.pop(){
            self.nodes[n as usize]=Node{next,..cand.to_node()};
            n
        } else{
            let n=self.nodes.len() as uint;
            assert!(n!=0,"uintのサイズが足りないよ");
            self.nodes.push(Node{next,..cand.to_node()});
            n
        };

        if next!=!0{
            self.nodes[next as usize].prev=new;
        }
        self.nodes[cand.parent as usize].child=new;
        
        self.next_leaf.push(new);
    }

    fn del_node(&mut self,mut idx:uint){
        loop{
            self.free.push(idx);
            let Node{prev,next,parent,..}=self.nodes[idx as usize];
            assert_ne!(parent,!0,"全てのノードを消そうとしています");

            if prev&next==!0{
                idx=parent;
                continue;
            }

            if prev!=!0{
                self.nodes[prev as usize].next=next;
            }
            else{
                self.nodes[parent as usize].child=next;
            }
            if next!=!0{
                self.nodes[next as usize].prev=prev;
            }
            
            break;
        }
    }

    fn dfs(&mut self,input:&Input,cands:&mut Vec<Cand>,single:bool){
        if self.nodes[self.cur_node].child==!0{
            self.append_cands(input,self.cur_node,cands);
            return;
        }

        let node=self.cur_node;
        let mut child=self.nodes[node].child;
        let next_single=single&(self.nodes[child as usize].next==!0);

        // let prev_state=self.state.clone();
        loop{
            self.cur_node=child as usize;
            self.state.apply(&self.nodes[child as usize]);
            self.dfs(input,cands,next_single);

            if !next_single{
                self.state.revert(&self.nodes[child as usize]);
                // assert!(prev_state==self.state);
            }
            child=self.nodes[child as usize].next;
            if child==!0{
                break;
            }
        }
        
        if !next_single{
            self.cur_node=node;
        }
    }

    fn no_dfs(&mut self,input:&Input,cands:&mut Vec<Cand>){
        loop{
            let Node{next,child,..}=self.nodes[self.cur_node];
            if next==!0 || child==!0{
                break;
            }
            self.cur_node=child as usize;
            self.state.apply(&self.nodes[self.cur_node]);
        }

        let root=self.cur_node;
        loop{
            let child=self.nodes[self.cur_node].child;
            if child==!0{
                self.append_cands(input,self.cur_node,cands);
                loop{
                    if self.cur_node==root{
                        return;
                    }
                    let node=&self.nodes[self.cur_node];
                    self.state.revert(&node);
                    if node.next!=!0{
                        self.cur_node=node.next as usize;
                        self.state.apply(&self.nodes[self.cur_node]);
                        break;
                    }
                    self.cur_node=node.parent as usize;
                }
            }
            else{
                self.cur_node=child as usize;
                self.state.apply(&self.nodes[self.cur_node]);
            }
        }
    }

    fn enum_cands(&mut self,input:&Input,cands:&mut Vec<Cand>){
        // self.dfs(input,cands,true);
        self.no_dfs(input,cands);
    }

    fn update(&mut self,cands:impl Iterator<Item=Cand>){
        self.next_leaf.clear();
        for cand in cands{
            self.add_node(cand);
        }

        for i in 0..self.leaf.len(){
            let n=self.leaf[i];
            if self.nodes[n as usize].child==!0{
                self.del_node(n);
            }
        }

        std::mem::swap(&mut self.leaf,&mut self.next_leaf);
    }

    fn restore(&self,mut idx:uint)->Vec<u8>{
        let mut ret=vec![];
        loop{
            let Node{op,parent,..}=self.nodes[idx as usize];
            if parent==!0{
                break;
            }
            ret.push(op);
            idx=parent;
        }
        
        ret.reverse();
        ret
    }

    fn append_cands(&self,input:&Input,idx:usize,cands:&mut Vec<Cand>){
        let node=&self.nodes[idx];
        assert_eq!(node.child,!0);

        todo!();
    }

    fn solve(&mut self,input:&Input)->Vec<u8>{
        use std::cmp::Reverse;
        let M=MAX_WIDTH;
    
        let mut cands:Vec<Cand>=vec![];
        let mut set=rustc_hash::FxHashSet::default();
        
        for t in 0..TURN{
            if t!=0{
                let M0=(M as f64*2.).round() as usize;
                if cands.len()>M0{
                    cands.select_nth_unstable_by_key(M0,|a|Reverse(a.eval_score));
                    cands.truncate(M0);
                }
                
                cands.sort_unstable_by_key(|a|Reverse(a.eval_score));
                set.clear();

                self.update(cands.drain(..).filter(|cand|
                    set.insert(cand.hash)
                ).take(M));
            }
            
            cands.clear();
            self.enum_cands(input,&mut cands);
            assert!(!cands.is_empty());
        }
    
        let best=cands.into_iter().max_by_key(|a|a.raw_score(input)).unwrap();
        eprintln!("score = {}",best.raw_score(input));
        
        let mut ret=self.restore(best.parent);
        ret.push(best.op);
    
        ret
    }
}
}

// #CAP(fumin::modint)
pub mod fumin {
pub mod grid_v {
#![allow(dead_code)]
use std::{ops::{Index, IndexMut}, cmp::Reverse};

use crate::{common::*, chmin};
use super::pt::{Pt, Dir};


#[derive(Debug, Clone)]
pub struct GridV<T> {
    g: Vec<T>,
    h: us,
    w: us,
}

impl<T: Clone> GridV<T> {
    pub fn with_default(h: us, w: us, v: T) -> Self {
        Self { g: vec![v; h * w], h, w, }
    }
    pub fn is_in_p<N: IntoT<us>>(&self, p: Pt<N>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t<N: IntoT<us>>(&self, t: (N, N)) -> bool { t.0.into_t() < self.h && t.1.into_t() < self.w }
}

impl<T: Clone + Default> GridV<T> {
    pub fn new(h: us, w: us) -> Self {
        Self { g: vec![T::default(); h * w], h, w, }
    }
}

impl ToString for GridV<char> {
    fn to_string(&self) -> String {
        let mut ret = "".to_owned();
        for i in 0..self.h {
            ret.push_str(format!("{}: ", i%10).as_str()); // line number
            ret.push_str(self[i].str().as_str());
            ret.push('\n');
        }
        ret

    }
}

impl<T, N: IntoT<us>> Index<N> for GridV<T> {
    type Output = [T];
    fn index(&self, i: N) -> &Self::Output {
        let idx = i.into_t() * self.w;
        &self.g[idx..idx+self.w]
    }
}
impl<T, N: IntoT<us>> IndexMut<N> for GridV<T> {
    fn index_mut(&mut self, i: N) -> &mut Self::Output {
        let idx = i.into_t() * self.w;
        &mut self.g[idx..idx+self.w]
    }
}

impl<T, N: IntoT<us>> Index<(N,N)> for GridV<T> {
    type Output = T;
    fn index(&self, index: (N,N)) -> &Self::Output { &self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> IndexMut<(N,N)> for GridV<T> {
    fn index_mut(&mut self, index: (N,N)) -> &mut Self::Output { &mut self[index.0.into_t()][index.1.into_t()] }
}
impl<T, N: IntoT<us>> Index<Pt<N>> for GridV<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: IntoT<us>> IndexMut<Pt<N>> for GridV<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T: Clone> From<&Vec<Vec<T>>> for GridV<T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        let (h, w) = (value.len(), value[0].len());
        GridV{ g: value.iter().cloned().flatten().cv(), h, w }
    }
}

pub struct ShortestPath {
    pub start: Pt<us>,
    pub dist: GridV<i64>,
    pub prev: GridV<Pt<us>>,
}

impl ShortestPath {
    pub fn restore_shortest_path(&self, mut t: Pt<us>) -> Vec<Pt<us>> {
        let mut ps = vec![];
        while t != Pt::<us>::INF { ps.push(t); t = self.prev[t]; }
        ps.reverse();
        assert!(ps[0] == self.start);
        ps
    }
}

impl GridV<char> {
    // まだあまり動かしてないので、そのうちテスト必要
    pub fn bfs(&self, s: Pt<us>) -> ShortestPath {
        let mut que = deque::new();
        let mut ret = ShortestPath {
            start: s,
            dist: GridV::with_default(self.h, self.w, i64::INF),
            prev: GridV::with_default(self.h, self.w, Pt::<us>::INF),
        };
        que.push_back(s);
        ret.dist[s] = 0;
        while let Some(v) = que.pop_front() {
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if self.is_in_p(nv) && self[nv]!='#' && ret.dist[nv]==i64::INF {
                    ret.dist[nv] = ret.dist[v]+1;
                    ret.prev[nv] = v;
                    que.push_back(nv);
                }
            }
        }
        ret
    }
}


pub trait CellTrait {
    fn cost(&self, d: Dir) -> Option<i64>;
}

impl<T: CellTrait> GridV<T> {
    pub fn dijkstra(&self, s: Pt<us>) -> ShortestPath {
        type P = Pt<us>;
        let mut ret = ShortestPath {
            start: s,
            dist: GridV::with_default(self.h,self.w,i64::INF),
            prev: GridV::with_default(self.h,self.w,P::INF),
        };
        let mut q = bheap::new();
        q.push(Reverse((0,s)));
        ret.dist[s] = 0;
        while let Some(Reverse((cost, v))) = q.pop() {
            if ret.dist[v] < cost { continue; }
            for d in Dir::VAL4 {
                let Some(c) = self[v].cost(d) else { continue; };
                let nv = v.next(d);
                let nc = cost + c;
                if chmin!(ret.dist[nv], nc) {
                    q.push(Reverse((nc, nv)));
                    ret.prev[nv] = v;
                }
            }
        }
        ret
    }
}
}
pub mod pt {
#![allow(dead_code)]
use std::{*, ops::*, iter::Sum};
use itertools::{iproduct, Itertools};
use num_traits::Signed;
use rand::Rng;

use crate::{common::*, enrich_enum, count};


// Pt
#[derive(Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N:fmt::Display> fmt::Debug for Pt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl<N> Pt<N> {
    pub fn new(x: impl IntoT<N>, y: impl IntoT<N>) -> Self { Pt{x:x.into_t(), y:y.into_t()} }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
}
impl<N: SimplePrimInt> Pt<N> {
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn on(self, h: Range<N>, w: Range<N>) -> bool { h.contains(&self.x) && w.contains(&self.y) }
    pub fn manhattan_distance(self, p: Pt<N>) -> N { abs_diff(self.x, p.x) + abs_diff(self.y, p.y) }
}

impl<N: SimplePrimInt+FromT<i64>+ToF64> Pt<N> {
    pub fn norm(self) -> f64 { self.norm2().f64().sqrt() }
}
impl<N: Wrapping> Wrapping for Pt<N> {
    fn wrapping_add(self, a: Self) -> Self { Self::of(self.x.wrapping_add(a.x), self.y.wrapping_add(a.y)) }
}
impl Pt<us> {
    pub fn wrapping_sub(self, a: Self) -> Self { Self::of(self.x.wrapping_sub(a.x), self.y.wrapping_sub(a.y)) }
    pub fn wrapping_mul(self, a: Self) -> Self { Self::of(self.x.wrapping_mul(a.x), self.y.wrapping_mul(a.y)) }
    pub fn next(self, d: Dir) -> Self { self.wrapping_add(d.p()) }
    pub fn iter_next_4d(self) -> impl Iterator<Item=Self> { Dir::VAL4.iter().map(move|&d|self.next(d)) }
    pub fn iter_next_8d(self) -> impl Iterator<Item=Self> { Dir::VALS.iter().map(move|&d|self.next(d)) }
    pub fn prev(self, d: Dir) -> Self { self.wrapping_sub(d.p()) }
    pub fn iter(rx: Range<us>, ry: Range<us>) -> impl Iterator<Item=Self> { iproduct!(rx, ry).map(|t|Self::from(t)) }
    fn to_dir(a:Self, b:Self) -> Option<Dir> {
        if a == b { return None; }
        if a.x == b.x {
            if a.y < b.y { Some(Dir::R) } else { Some(Dir::L) }
        } else if a.y == b.y {
            if a.x < b.x { Some(Dir::D) } else { Some(Dir::U) }
        } else {
            unreachable!("a and b are distant from each other. a={}, b={}", a, b);
        }
    }
    pub fn to_dirs(rt:&Vec<Self>) -> Vec<Dir> {
        rt.iter().tuple_windows()
            .filter_map(|(&a,&b)|Self::to_dir(a,b))
            .cv()
    }

    pub fn move_xy(a:Self, b:Self) -> Vec<Self> {
        // aからbに縦->横の順に進む
        Self::move_impl(a, b, Self::new(b.x, a.y))
    }
    pub fn move_yx(a:Self, b:Self) -> Vec<Self> {
        // aからbに横->縦の順に進む
        Self::move_impl(a, b, Self::new(a.x, b.y))
    }
    fn move_impl(a:Self, b:Self, corner:Self) -> Vec<Self> {
        let mut ret = vec![a];
        let mut v = a;
        for (x, y) in [(a, corner), (corner, b)] {
            let d = Self::to_dir(x, y).unwrap();
            for _ in 0..x.manhattan_distance(y) { v = v.next(d); ret.push(v); }
        }
        ret
    }
}

impl<T: Inf> Inf for Pt<T> {
    const INF: Self  = Pt::<T>{x: T::INF,  y: T::INF};
    const MINF: Self = Pt::<T>{x: T::MINF, y: T::MINF};
}

impl<N: Copy + Signed> Pt<N> {
    // 外積 (ベクトルself, vからなる平行四辺形の符号付面積)
    pub fn cross(&self, v: Pt<N>) -> N {
        self.x * v.y - self.y * v.x
    }

    // couter cross wise (ベクトルself, vが反時計回りかどうか)
    pub fn ccw(&self, v: Pt<N>) -> i32 {
        let a = self.cross(v);
        if a.is_positive() { 1 } // ccw
        else if a.is_negative() { -1 } // cw
        else { 0 } // colinear
    }
}

pub type Radian = f64;
impl Pt<f64> {
    pub fn rot(self, r: Radian) -> Pt<f64> {
        let (x, y) = (self.x, self.y);
        Self::new(r.cos()*x-r.sin()*y, r.sin()*x+r.cos()*y) // 反時計回りにr度回転(rはradian)
    }
}
impl<N: SimplePrimInt+fmt::Display> fmt::Display  for Pt<N> { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{} {}", self.x, self.y) } }
impl<N: SimplePrimInt+fmt::Display> Fmt           for Pt<N> { fn fmt(&self) -> String { format!("{} {}", self.x, self.y) } }
impl<N: AddAssign<N>+Copy> AddAssign<Pt<N>> for Pt<N> { fn add_assign(&mut self, rhs: Pt<N>) { self.x += rhs.x; self.y += rhs.y; } }
impl<N: SubAssign<N>+Copy> SubAssign<Pt<N>> for Pt<N> { fn sub_assign(&mut self, rhs: Pt<N>) { self.x -= rhs.x; self.y -= rhs.y; } }
impl<N: AddAssign<N>+Copy> AddAssign<N>     for Pt<N> { fn add_assign(&mut self, rhs: N) { self.x += rhs; self.y += rhs; } }
impl<N: SubAssign<N>+Copy> SubAssign<N>     for Pt<N> { fn sub_assign(&mut self, rhs: N) { self.x -= rhs; self.y -= rhs; } }
impl<N: MulAssign<N>+Copy> MulAssign<N>     for Pt<N> { fn mul_assign(&mut self, rhs: N) { self.x *= rhs; self.y *= rhs; } }
impl<N: DivAssign<N>+Copy> DivAssign<N>     for Pt<N> { fn div_assign(&mut self, rhs: N) { self.x /= rhs; self.y /= rhs; } }
impl<N: RemAssign<N>+Copy> RemAssign<N>     for Pt<N> { fn rem_assign(&mut self, rhs: N) { self.x %= rhs; self.y %= rhs; } }
impl<N: AddAssign<N>+Copy> Add<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: Pt<N>) -> Self::Output { self += rhs; self } }
impl<N: SubAssign<N>+Copy> Sub<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: Pt<N>) -> Self::Output { self -= rhs; self } }
impl<N: AddAssign<N>+Copy> Add<N>           for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: N) -> Self::Output { self += rhs; self } }
impl<N: SubAssign<N>+Copy> Sub<N>           for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: N) -> Self::Output { self -= rhs; self } }
impl<N: MulAssign<N>+Copy> Mul<N>           for Pt<N> { type Output = Pt<N>; fn mul(mut self, rhs: N) -> Self::Output { self *= rhs; self } }
impl<N: DivAssign<N>+Copy> Div<N>           for Pt<N> { type Output = Pt<N>; fn div(mut self, rhs: N) -> Self::Output { self /= rhs; self } }
impl<N: RemAssign<N>+Copy> Rem<N>           for Pt<N> { type Output = Pt<N>; fn rem(mut self, rhs: N) -> Self::Output { self %= rhs; self } }
impl<N: SimplePrimInt+FromT<is>> Neg        for Pt<N> { type Output = Pt<N>; fn neg(mut self) -> Self::Output { self *= N::from_t(-1); self } }
impl<N: SimplePrimInt+Default> Sum          for Pt<N> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::default(), |a, b| a + b) } }

impl<N: SimplePrimInt+FromT<is>+proconio::source::Readable<Output=N>+IntoT<N>> proconio::source::Readable for Pt<N> {
    type Output = Pt<N>;
    fn read<R: io::BufRead, S: proconio::source::Source<R>>(source: &mut S) -> Self::Output {
        Pt::new(N::read(source), N::read(source))
    }
}
impl<T> From<(T, T)> for Pt<T> {
    fn from(t: (T, T)) -> Self { Self::of(t.0, t.1) }
}

enrich_enum! {
    pub enum Dir {
        R, L, D, U, RU, RD, LD, LU,
    }
}

impl Dir {
    pub const C4: [char; 4] = ['R', 'L', 'D', 'U'];
    pub const VAL4: [Self; 4] = [Self::R,Self::L,Self::D,Self::U];
    pub const P8: [Pt<us>; 8] = [
        Pt::<us>{x:0,y:1},Pt::<us>{x:0,y:!0},Pt::<us>{x:1,y:0},Pt::<us>{x:!0,y:0},
        Pt::<us>{x:1,y:1},Pt::<us>{x:1,y:!0},Pt::<us>{x:!0,y:1},Pt::<us>{x:!0,y:!0},
        ];
    pub const REV8: [Self; 8] = [
        Self::L,Self::R,Self::U,Self::D,
        Self::LD,Self::LU,Self::RU,Self::RD,
        ];
    pub const RROT90: [Self; 4] = [Self::D,Self::U,Self::L,Self::R];
    pub const LROT90: [Self; 4] = [Self::U,Self::D,Self::R,Self::L];
    pub const RROT45: [Self; 8] = [
        Self::RD,Self::LU,Self::LD,Self::RU,
        Self::R,Self::D,Self::L,Self::U,
        ];
    pub const LROT45: [Self; 8] = [
        Self::RU,Self::LD,Self::RD,Self::LU,
        Self::U,Self::R,Self::D,Self::L,
        ];

    #[inline] pub const fn c(self) -> char   { Self::C4[self.id()] }
    #[inline] pub const fn p(self) -> Pt<us> { Self::P8[self.id()] }
    #[inline] pub const fn rev(self) -> Self { Self::REV8[self.id()] }
    #[inline] pub const fn rrot90(self) -> Self { Self::RROT90[self.id()] }
    #[inline] pub const fn lrot90(self) -> Self { Self::LROT90[self.id()] }
    #[inline] pub const fn rrot45(self) -> Self { Self::RROT45[self.id()] }
    #[inline] pub const fn lrot45(self) -> Self { Self::LROT45[self.id()] }

    #[inline] pub fn dir(a: Pt<us>, b: Pt<us>) -> Self { (b.wrapping_sub(a)).into() } // a -> b
    #[inline] pub fn rng4(rng: &mut impl rand_core::RngCore) -> Dir { Self::VALS[rng.gen_range(0..4)] }
    #[inline] pub fn rng8(rng: &mut impl rand_core::RngCore) -> Dir { Self::VALS[rng.gen_range(0..8)] }

}
impl From<Pt<us>> for Dir { fn from(value: Pt<us>) -> Self { Self::P8.pos(&value).unwrap().into() } }
impl From<char>   for Dir { fn from(value: char) -> Self { Self::C4.pos(&value).unwrap().into() } }


}
pub mod enrich_enum {
#![allow(dead_code)]

#[macro_export]
macro_rules! count {
     () => (0usize);
     ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

#[macro_export]
macro_rules! enrich_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($e:ident,)* }) => {
        static_assertions::const_assert!(count!($($e,)*) <= 32);

        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
        $(#[$meta])*
        $vis enum $name {
            $($e,)*
        }
        impl $name {
            pub const VALS: [Self; count!($($e)*)] = [$(Self::$e,)*];
            #[inline] pub const fn id(self) -> us { self as us }
            #[inline] pub const fn b(self) -> u32 { 1<<self.id() }
            #[inline] pub const fn is_or(self, b: u32) -> bool { self.b() & b > 0 }
            #[inline] pub const fn from_id(id: us) -> Self { Self::VALS[id] }
            #[inline] pub const fn from_bit(b: u32) -> Self {
                assert!(b.count_ones() <= 1);
                Self::from_id(b.trailing_zeros() as usize + 1)
            }
        }
        impl BitOr for $name {
            type Output = u32;
            fn bitor(self, rhs: Self) -> Self::Output { self.b() | rhs.b() }
        }

        impl From<us>  for $name { fn from(value: us) -> Self { Self::from_id(value) } }
        impl From<u32> for $name { fn from(value: u32) -> Self { Self::from_bit(value) } }
    };
}
}
pub mod fast_bit_set {
#![allow(dead_code)]

use crate::common::us;
use super::pt;

type P = pt::Pt<us>;

// フラグのチェックとclearがO(1)でできる
pub struct FastBitSet {
    bs: Vec<u32>,
    id: u32,
}

impl FastBitSet {
    pub fn new(n: usize) -> Self { Self { bs: vec![0; n], id: 1, } }
    pub fn clear(&mut self) { self.id += 1; }
    pub fn set(&mut self, i: usize, f: bool) { self.bs[i] = if f { self.id } else { 0 }; }
}

impl std::ops::Index<usize> for FastBitSet {
    type Output = bool;
    fn index(&self, i: usize) -> &bool {
        if self.bs[i] == self.id { &true } else { &false }
    }
}

pub struct FastBitSet2d {
    bs: Vec<u32>,
    id: u32,
    pub h: us,
    pub w: us,
}

impl FastBitSet2d {
    pub fn new(h: us, w:us) -> Self { Self { bs: vec![0; h*w], id: 1, h, w, } }
    pub fn clear(&mut self) { self.id += 1; }
    pub fn set(&mut self, v: P, f: bool) {
        let idx = self.idx(v);
        self.bs[idx] = if f { self.id } else { 0 };
    }
    fn idx(&self, v: P) -> us { v.x * self.w + v.y }
}

impl std::ops::Index<P> for FastBitSet2d {
    type Output = bool;
    fn index(&self, v: P) -> &bool {
        if self.bs[self.idx(v)] == self.id { &true } else { &false }
    }
}

}
}

pub mod common {
#![allow(dead_code, unused_imports, unused_macros, non_snake_case, non_camel_case_types)]
use std::{*, ops::*, collections::*, iter::{Sum, FromIterator}};
use itertools::Itertools;
use ::num::{One, Zero};

pub type us        = usize;
pub type is        = isize;
pub type us1       = proconio::marker::Usize1;
pub type is1       = proconio::marker::Isize1;
pub type chars     = proconio::marker::Chars;
pub type bytes     = proconio::marker::Bytes;
pub type Str       = String;
pub type map<K,V>  = HashMap<K,V>;
pub type bmap<K,V> = BTreeMap<K,V>;
pub type set<V>    = HashSet<V>;
pub type bset<V>   = BTreeSet<V>;
pub type bheap<V>  = BinaryHeap<V>;
pub type deque<V>  = VecDeque<V>;

pub trait FromT<T> { fn from_t(t: T) -> Self; }
pub trait IntoT<T> { fn into_t(self) -> T; }

// PrimNum
pub trait SimplePrimInt:
        Copy
        + PartialOrd<Self>
        + Add<Output=Self>
        + Sub<Output=Self>
        + Mul<Output=Self>
        + Div<Output=Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Default
        + PartialEq
        + Zero
        + One
{
    fn two() -> Self { Self::one() + Self::one() }
}
 
pub trait ExPrimInt: SimplePrimInt
        + Rem<Output=Self>
        + RemAssign
        + FromT<us>
{}


#[macro_export] macro_rules! impl_prim_num {
    ($($t:ty),*) => {$(
        impl SimplePrimInt for $t { }
        impl ExPrimInt     for $t { }
        impl FromT<us>   for $t { fn from_t(n: us) -> Self { n as $t } }
        impl FromT<is>   for $t { fn from_t(n: is) -> Self { n as $t } }
        impl IntoT<us>   for $t { fn into_t(self)  -> us   { self as us  } }
        impl IntoT<is>   for $t { fn into_t(self)  -> is   { self as is  } }
        impl IntoT<f64>  for $t { fn into_t(self)  -> f64  { self as f64 } }
        impl IntoT<u8>   for $t { fn into_t(self)  -> u8   { self as u8  } }
        impl IntoT<u32>  for $t { fn into_t(self)  -> u32  { self as u32 } }
        impl IntoT<u64>  for $t { fn into_t(self)  -> u64  { self as u64 } }
        impl IntoT<i32>  for $t { fn into_t(self)  -> i32  { self as i32 } }
        impl IntoT<i64>  for $t { fn into_t(self)  -> i64  { self as i64 } }
        impl IntoT<char> for $t { fn into_t(self)  -> char { (self as u8) as char } }
    )*}
}
impl_prim_num! {isize, i8, i32, i64, usize, u8, u32, u64, f32, f64}

pub trait ToUs   { fn us(self) -> us; }
pub trait ToIs   { fn is(self) -> is; }
pub trait ToI64  { fn i64(self) -> i64; }
pub trait ToF64  { fn f64(self) -> f64; }
pub trait ToU8   { fn u8(self) -> u8; }
pub trait ToU32  { fn u32(self) -> u32; }
pub trait ToI32  { fn i32(self) -> i32; }
pub trait ToChar { fn char(self) -> char; }

impl<T: IntoT<us>>   ToUs   for T { fn us(self)   -> us   { self.into_t() } }
impl<T: IntoT<is>>   ToIs   for T { fn is(self)   -> is   { self.into_t() } }
impl<T: IntoT<i64>>  ToI64  for T { fn i64(self)  -> i64  { self.into_t() } }
impl<T: IntoT<f64>>  ToF64  for T { fn f64(self)  -> f64  { self.into_t() } }
impl<T: IntoT<u8>>   ToU8   for T { fn u8(self)   -> u8   { self.into_t() } }
impl<T: IntoT<u32>>  ToU32  for T { fn u32(self)  -> u32  { self.into_t() } }
impl<T: IntoT<i32>>  ToI32  for T { fn i32(self)  -> i32  { self.into_t() } }
impl<T: IntoT<char>> ToChar for T { fn char(self) -> char { self.into_t() } }
impl IntoT<us>   for char  { fn into_t(self) -> us   { self as us } }
impl IntoT<is>   for char  { fn into_t(self) -> is   { self as is } }
impl IntoT<u8>   for char  { fn into_t(self) -> u8   { self as u8 } }
impl IntoT<u32>  for char  { fn into_t(self) -> u32  { self as u32 } }
impl IntoT<i32>  for char  { fn into_t(self) -> i32  { self as i32 } }
impl IntoT<u64>  for char  { fn into_t(self) -> u64  { self as u64 } }
impl IntoT<i64>  for char  { fn into_t(self) -> i64  { self as i64 } }
impl IntoT<char> for char  { fn into_t(self) -> char { self } }
impl IntoT<char> for &char { fn into_t(self) -> char { *self } }

pub trait Inf {
    const INF: Self;
    const MINF: Self;
}
impl Inf for us  {
    const INF: Self = std::usize::MAX / 4;
    const MINF: Self = 0;
}
impl Inf for is {
    const INF: Self = std::isize::MAX / 4;
    const MINF: Self = -Self::INF;
}
impl Inf for i64 {
    const INF: Self = std::i64::MAX / 4;
    const MINF: Self = -Self::INF;
}

pub trait Wrapping {
    fn wrapping_add(self, a: Self) -> Self;
}
impl Wrapping for us  { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for is  { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for i64 { fn wrapping_add(self, a: Self) -> Self { self.wrapping_add(a) } }

// Utilities
#[macro_export] macro_rules! or    { ($cond:expr;$a:expr,$b:expr) => { if $cond { $a } else { $b } }; }
#[macro_export] macro_rules! chmax { ($a:expr,$b:expr) => { { let v = $b; if $a < v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! chmin { ($a:expr,$b:expr) => { { let v = $b; if $a > v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! add   { ($a:expr,$b:expr) => { { let v = $b; $a += v; } } }
#[macro_export] macro_rules! sub   { ($a:expr,$b:expr) => { { let v = $b; $a -= v; } } }
#[macro_export] macro_rules! mul   { ($a:expr,$b:expr) => { { let v = $b; $a *= v; } } }
#[macro_export] macro_rules! div   { ($a:expr,$b:expr) => { { let v = $b; $a /= v; } } }
#[macro_export] macro_rules! rem   { ($a:expr,$b:expr) => { { let v = $b; $a %= v; } } }

pub fn abs_diff<T:PartialOrd+Sub<Output=T>>(n1: T, n2: T) -> T { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
pub fn floor<N: SimplePrimInt>(a: N, b: N) -> N { a / b }
pub fn ceil<N: SimplePrimInt>(a: N, b: N) -> N { (a + b - N::one()) / b }
pub fn asc <T:Ord>(a: &T, b: &T) -> cmp::Ordering { a.cmp(b) }
pub fn desc<T:Ord>(a: &T, b: &T) -> cmp::Ordering { b.cmp(a) }
pub fn to_int<T:Zero+One>(a: bool) -> T { if a { T::one() } else { T::zero() } }
pub fn min_max<T: Ord+Copy>(a: T, b: T) -> (T, T) { (cmp::min(a,b), cmp::max(a,b)) }
pub fn bin_search<T: ExPrimInt+Shr<Output=T>>(mut ok: T, mut ng: T, f: impl Fn(T)->bool) -> T {
    while abs_diff(ok, ng) > T::one() {
        let m = (ok + ng) >> T::one();
        if f(m) { ok = m; } else { ng = m; }
    }
    ok
}

pub trait IterTrait : Iterator {
    fn counts<C>(&mut self) -> CountIter<Self, C>
        where
            Self: Sized,
            Self::Item: hash::Hash + Eq + Clone,
            C: Eq + AddAssign<C> + One + Default + Copy,
    {
        CountIter::new(self)
    }
    fn grouping_to_bmap<'a, K:Ord+Clone, V>(&'a mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> bmap<K, Vec<V>> {
        self.fold(bmap::<_,_>::new(), |mut m, x| { m.entry(get_key(&x)).or_default().push(get_val(&x)); m })
    }
    fn grouping_to_map<K:Eq+hash::Hash+Clone, V>(&mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> map<K, Vec<V>> {
        self.fold(map::<_,_>::new(), |mut m, x| { m.entry(get_key(&x)).or_default().push(get_val(&x)); m })
    }
    fn cv(&mut self) -> Vec<Self::Item> { self.collect_vec() }

    fn shuffled(self, rng: &mut impl rand_core::RngCore) -> vec::IntoIter<Self::Item> where Self: Sized {
        use rand::seq::SliceRandom;
        let mut v = Vec::from_iter(self);
        v.shuffle(rng);
        v.into_iter()
    }

}

pub struct CountIter<I: Iterator, C> {
    nexts: deque<(I::Item, C)>,
}
impl<I: Iterator, C> CountIter<I, C>
    where
        I::Item: hash::Hash+Eq+Clone,
        C: Eq + AddAssign<C> + One + Default + Copy,
        {
    pub fn new(iter: &mut I) -> Self {
        let mut cnt = map::new();
        let mut keys = Vec::new();
        while let Some(e) = iter.next() {
            *cnt.entry(e.clone()).or_default() += C::one();
            if cnt[&e] == C::one() { keys.push(e); }
        }
        let nexts = deque::from_iter(
            keys.into_iter().map(|k| { let c = cnt[&k]; (k,c) } ));
        Self { nexts }
    }
}
impl<I: Iterator, C> Iterator for CountIter<I, C> where I::Item: hash::Hash + Eq + Clone {
    type Item = (I::Item, C);
    fn next(&mut self) -> Option<Self::Item> {
        self.nexts.pop_front()
    }
}

pub trait CharIterTrait<T: IntoT<char>> : Iterator<Item=T> {
    fn cstr(&mut self) -> String { self.map(|c|c.into_t()).collect::<Str>() }
}
pub trait HashIterTrait : Iterator where Self::Item: Eq+hash::Hash {
    fn cset(&mut self) -> set<Self::Item> { self.collect::<set<_>>() }
}
pub trait OrdIterTrait : Iterator where Self::Item: Ord {
    fn cbset(&mut self) -> bset<Self::Item> { self.collect::<bset<_>>() }
}
pub trait PairHashIterTrait<T, U> : Iterator<Item=(T,U)> where T: Eq+hash::Hash {
    fn cmap(&mut self) -> map<T, U> { self.collect::<map<_,_>>() }
}
pub trait PairOrdIterTrait<T, U> : Iterator<Item=(T,U)> where T: Ord {
    fn cbmap(&mut self) -> bmap<T, U> { self.collect::<bmap<_,_>>() }
}

impl<I> IterTrait     for I where I: Iterator { }
impl<I, T: IntoT<char>> CharIterTrait<T> for I where I: Iterator<Item=T> { }
impl<I> HashIterTrait for I where I: Iterator, Self::Item: Eq+hash::Hash { }
impl<I> OrdIterTrait  for I where I: Iterator, Self::Item: Ord { }
impl<I, T, U> PairHashIterTrait<T, U> for I where I: Iterator<Item=(T,U)>, T: Eq+hash::Hash { }
impl<I, T, U> PairOrdIterTrait<T, U>  for I where I: Iterator<Item=(T,U)>, T: Ord { }


// Vec
pub trait VecCount<T> { fn count(&self, f: impl FnMut(&T)->bool) -> us; }
impl<T> VecCount<T> for [T] { fn count(&self, mut f: impl FnMut(&T)->bool) -> us { self.iter().filter(|&x|f(x)).count() } }

pub trait VecMax<T> { fn vmax(&self) -> T; }
impl<T:Clone+Ord> VecMax<T> for [T] { fn vmax(&self) -> T  { self.iter().cloned().max().unwrap() } }

pub trait VecMin<T> { fn vmin(&self) -> T; }
impl<T:Clone+Ord> VecMin<T> for [T] { fn vmin(&self) -> T  { self.iter().cloned().min().unwrap() } }

pub trait VecSum<T> { fn sum(&self) -> T; }
impl<T:Clone+Sum<T>> VecSum<T> for [T] { fn sum(&self)  -> T  { self.iter().cloned().sum::<T>() } }

pub trait VecStr<T> { fn str(&self) -> Str; }
impl<T:ToString> VecStr<T> for [T] { fn str(&self)  -> Str { self.iter().map(|x|x.to_string()).collect::<Str>() } }

pub trait VecMap<T> { fn map<U>(&self, f: impl FnMut(&T)->U) -> Vec<U>; }
impl<T> VecMap<T> for [T] { fn map<U>(&self, mut f: impl FnMut(&T)->U) -> Vec<U> { self.iter().map(|x|f(x)).cv() } }

pub trait VecPos<T> { fn pos(&self, t: &T) -> Option<us>; }
impl<T:Eq> VecPos<T> for [T] { fn pos(&self, t: &T) -> Option<us> { self.iter().position(|x|x==t) } }

pub trait VecRpos<T> { fn rpos(&self, t: &T) -> Option<us>; }
impl<T:Eq> VecRpos<T> for [T] { fn rpos(&self, t: &T) -> Option<us> { self.iter().rposition(|x|x==t) } }

pub trait VecSet<T> { fn set(&mut self, i: us, t: T) -> T; }
impl<T> VecSet<T> for [T] { fn set(&mut self, i: us, mut t: T) -> T { std::mem::swap(&mut self[i], &mut t); t } }

// Deque
pub trait DequePush<T> { fn push(&mut self, t: T); }
impl<T> DequePush<T> for VecDeque<T> { fn push(&mut self, t: T) { self.push_back(t); } }

pub trait DequePop<T> { fn pop(&mut self) -> Option<T>; }
impl<T> DequePop<T> for VecDeque<T> { fn pop(&mut self) -> Option<T> { self.pop_back() } }

pub trait Identify {
    type Ident;
    fn ident(&self) -> Self::Ident;
    fn ident_by(&self, s: &str) -> Self::Ident;
}

impl<T: IntoT<u8> + Copy> Identify for T {
    type Ident = us;

    fn ident(&self) -> us {
        let c = self.into_t();
        if b'0' <= c && c <= b'9'      { (c - b'0').us() }
        else if b'a' <= c && c <= b'z' { (c - b'a').us() }
        else if b'A' <= c && c <= b'Z' { (c - b'A').us() }
        else { 0 }
    }

    fn ident_by(&self, s: &str) -> us {
        let b = self.into_t();
        s.chars().position(|c|c==b.char()).expect("error")
    }
}
impl<T: Identify> Identify for [T] {
    type Ident = Vec<T::Ident>;
    fn ident(&self) -> Self::Ident { self.iter().map(|x|x.ident()).collect_vec() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.iter().map(|x|x.ident_by(s)).collect_vec() }
}
impl Identify for &str {
    type Ident = Vec<us>;
    fn ident(&self) -> Self::Ident { self.as_bytes().ident() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.as_bytes().ident_by(s) }
}
impl Identify for String {
    type Ident = Vec<us>;
    fn ident(&self) -> Self::Ident { self.as_bytes().ident() }
    fn ident_by(&self, s: &str) -> Self::Ident { self.as_bytes().ident_by(s) }
}

pub trait ToC: IntoT<u8> + Copy {
    fn to_c_by(self, ini: char) -> char { (ini.u8() + self.u8()).char() }
}
impl<T: IntoT<u8>+Copy> ToC for T {}

trait Joiner { fn join(self, sep: &str) -> String; }
impl<It: Iterator<Item=String>> Joiner for It { fn join(self, sep: &str) -> String { self.collect::<Vec<_>>().join(sep) } }

pub trait Fmt { fn fmt(&self) -> String; }
macro_rules! fmt_primitive { ($($t:ty),*) => { $(impl Fmt for $t { fn fmt(&self) -> String { self.to_string() }})* } }

fmt_primitive! {
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128,
    usize, isize, f32, f64, char, &str, String, bool
}

impl<T: Fmt> Fmt for [T]         { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for VecDeque<T> { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for set<T>      { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
impl<T: Fmt> Fmt for bset<T>     { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }

#[macro_export] macro_rules! fmt {
    ($a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(($a)), fmt!($($b),*)) }};
    ($a:expr)               => {{ ($a).fmt() }};

    (@debug $a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(@debug ($a)), fmt!(@debug $($b),*)) }};
    (@debug $a:expr)               => {{ format!("{:?}", ($a)) }};
}

#[macro_export] macro_rules! vprintln {
    ($a:expr) => { for x in &($a) { println!("{}", x); } };
}
#[macro_export] macro_rules! vprintsp {
    ($a:expr) => {
        {
            use itertools::Itertools;
            println!("{}", ($a).iter().join(" "));
        }
    }
}
#[macro_export] macro_rules! print_grid {
    ($a:expr) => { for v in &($a) { println!("{}", v.iter().collect::<Str>()); } };
}

#[macro_export]#[cfg(feature="local")] macro_rules! debug {
    ($($a:expr),*)    => { eprintln!("{}", fmt!(@debug  $($a),*)); };
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug {
    ($($a:expr),*)    => { };
}
#[macro_export]#[cfg(feature="local")] macro_rules! debug2d {
    ($a:expr) => { for v in &($a) { eprintln!("{:?}", v); } };
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug2d {
    ($a:expr) => { };
}

pub fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }
pub fn Yes(b: bool) -> &'static str { if b { "Yes" } else { "No" } }
pub fn YES(b: bool) -> &'static str { if b { "YES" } else { "NO" } }
pub fn no(b: bool) -> &'static str { yes(!b) }
pub fn No(b: bool) -> &'static str { Yes(!b) }
pub fn NO(b: bool) -> &'static str { YES(!b) }


}
