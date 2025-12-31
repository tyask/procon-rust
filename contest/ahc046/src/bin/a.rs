#![allow(unused_imports)]
use std::{*, collections::*, ops::*, cmp::*, iter::*};
use grid_v::GridV;
use itertools::iproduct;
use proconio::{input, fastout};
use common::*;
use fumin::*;
use pt::Dir;
use rand::{RngCore, SeedableRng};
use rand_core::block;
use rustc_hash::FxHashSet;
use time::Instant;

fn main() {
    solve();
}

const N: us = 20;
const M: us = 40;
const T: us = 2 * N * M;
type P = pt::Pt<us>;

#[derive(Debug, Clone)]
struct Input {
    st: Instant,
    n:us,
    m:us,
    t:Vec<P>,
    zob_pos: grid_v::GridV<u64>,
    zob_block: grid_v::GridV<u64>,
    zob_target: Vec<u64>,
}

impl Input {
    fn new(rng: &mut impl RngCore) -> Self {
        let st = Instant::now();
        input! {n:us,m:us,t:[P;m]}

        let mut zob_pos = GridV::new(n, n);
        let mut zob_block = GridV::new(n, n);
        let mut zob_target = vec![0; m];
        for (i,j) in iproduct!(0..n, 0..n) {
            zob_pos[i][j] = rng.next_u64();
            zob_block[i][j] = rng.next_u64();
        }
        for i in 0..m {
            zob_target[i] = rng.next_u64();
        }

        Self {
            st,
            n,
            m,
            t,
            zob_pos,
            zob_block,
            zob_target,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum ActionType {
    #[default] Move,
    Slide,
    Alter,
}

impl ActionType {
    fn to_char(self) -> char {
        match self {
            ActionType::Move => 'M',
            ActionType::Slide => 'S',
            ActionType::Alter => 'A',
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Op {
    t: ActionType,
    d: Dir,
    prev_pos: P,
    pos: P,
    reached: bool,
    hash: u64,
}

impl beam::NodeValue for Op {}

struct CandidateAppender {}
impl beam::CandidateAppender<Op, Context> for CandidateAppender {
    fn append_cands(&self, ctx: &mut Context, parent: &beam::Node<Op>, cands: &mut Vec<beam::Cand<Op>>) {
        Self::append_move(ctx, parent, cands);
        // Self::append_slide(ctx, parent, cands);
        // Self::append_alter(ctx, parent, cands);
    }
}

impl CandidateAppender {
    fn append_move(ctx: &Context, parent: &beam::Node<Op>, cands:&mut Vec<beam::Cand<Op>>) {
        // let node=&ctx.nodes[idx];
        let input = &ctx.input;
        // 隣に移動
        for d in Dir::VAL4 {
            let np = ctx.pos.next(d);
            if !ctx.block.is_in_p(np) || ctx.block[np] == 1 { continue; }

            let dist = ctx.dist(np, ctx.next_target());
            let reached = np == ctx.next_target();
            let tid = ctx.target_id + if reached { 1 } else { 0 };
            let eval_score =
                (tid - 1).i64() * (T.i64() + 1)
                - (ctx.terns.i64() + dist)
                ;

            // let mut hash = parent.value.hash;
            if ctx.terns == 0 {
                debug!("Cand", ctx.hash, parent.value.hash);
            }
            let mut hash = 0;
            hash ^= ctx.input.zob_pos[ctx.pos] ^ ctx.input.zob_pos[np];
            if reached {
                if tid - 1 < input.m { hash ^= input.zob_target[tid - 1]; }
                if tid < input.m { hash ^= input.zob_target[tid]; }
            }

            let cand = beam::Cand {
                parent: parent.id,
                score: eval_score,
                hash: ctx.hash ^ hash,
                is_end: tid == ctx.input.m,
                op: Op {t: ActionType::Move, d, prev_pos: ctx.pos, pos: P::INF, reached, hash,},
            };
            cands.push(cand);
        }
    }

    fn append_slide(ctx: &Context, parent: &beam::Node<Op>, cands:&mut Vec<beam::Cand<Op>>) {
        // slide
        let input = &ctx.input;
        for d in Dir::VAL4 {
            let mut np = ctx.pos;
            for _ in 0..N {
                let p2 = np.next(d);
                if !ctx.block.is_in_p(p2) || ctx.block[p2] == 1 { break; }
                np = p2;
            }
            if ctx.pos == np { continue; }

            let dist = ctx.dist(np, ctx.next_target());
            let reached = np == ctx.next_target();
            let tid = ctx.target_id + if reached { 1 } else { 0 };
            let eval_score = (tid - 1).i64() * (T.i64() + 1)
                - (ctx.terns.i64() + dist)
                ;
            let mut hash = 0;
            hash ^= input.zob_pos[ctx.pos] ^ input.zob_pos[np];
            if reached {
                if tid - 1 < input.m { hash ^= input.zob_target[tid - 1]; }
                if tid < input.m { hash ^= input.zob_target[tid]; }
            }

            let cand = beam::Cand {
                parent: parent.id,
                score: eval_score,
                hash: ctx.hash ^ hash,
                is_end: tid == input.m,
                op: Op {t: ActionType::Slide, d, prev_pos: ctx.pos, pos: np, reached, hash,},
            };
            cands.push(cand);
        }
    }

    fn append_alter(ctx: &mut Context, parent: &beam::Node<Op>, cands:&mut Vec<beam::Cand<Op>>) {
        for d in Dir::VAL4 {
            let np = ctx.pos.next(d);
            if !ctx.block.is_in_p(np) { continue; }

            ctx.alter_block(np);
            let dist = ctx.dist(ctx.pos, ctx.next_target());
            let tid = ctx.target_id;
            let eval_score = (tid - 1).i64() * (T.i64() + 1)
                - (ctx.terns.i64() + dist)
                ;
            let mut hash = 0;
            hash ^= ctx.input.zob_block[np];

            ctx.alter_block(np);
            let cand = beam::Cand {
                parent: parent.id,
                score: eval_score,
                hash: ctx.hash ^ hash,
                is_end: false,
                op: Op {t: ActionType::Alter, d, prev_pos: ctx.pos, pos: np, reached:false, hash, },
            };
            cands.push(cand);
        }
    }

}

// CONTEST(abcXXX-a)
#[fastout]
fn solve() {
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(0);
    let input = Input::new(&mut rng);
    let mut beam = {
        let cfg = beam::Config {
            max_width: 7000,
            tern: T,
        };
        let ctx = Context::new(&input);
        beam::BeamSearch::new(cfg, ctx, CandidateAppender{})
    };
    let ret = beam.solve();
    for a in ret {
        println!("{} {}", a.t.to_char(), a.d.c());
    }
}

fn solve0() {
    let mut rng = rand_pcg::Pcg64Mcg::from_os_rng();
    let input = Input::new(&mut rng);
    let mut beam = {
        let state = State::new(&input);
        let node = Node {
            parent:!0,
            child:!0,
            prev:!0,
            next:!0,
            action:Op::default(),
            hash: input.zob_pos[input.t[0]],
            score: 0,
            reached: false,
        };
        BeamSearch::new(state,node)
    };
    let ret = beam.solve(&input);
    for a in ret {
        println!("{} {}", a.t.to_char(), a.d.c());
    }
}

struct Context {
    input: Input,
    pos: P,
    block: grid_v::GridV<us>,
    block_h: Vec<bset<us>>,
    block_w: Vec<bset<us>>,
    target_id: us,
    targets: Vec<P>,
    terns: us,
    blocks: us,
    hash: u64,
}

impl Context {
    fn new(input: &Input) -> Self {
        let g = grid_v::GridV::with_default(input.n, input.n, 0);
        let block_h = vec![bset::default(); N];
        let block_w = vec![bset::default(); N];
        Self {
            input: input.clone(),
            pos: input.t[0],
            block: g,
            block_h,
            block_w,
            target_id: 1,
            targets: input.t.clone(),
            terns: 0,
            blocks: 0,
            hash: input.zob_pos[input.t[0]],
        }
    }
    fn alter_block(&mut self, p: P) {
        if self.block[p] == 0 {
            self.block_h[p.x].insert(p.y);
            self.block_w[p.y].insert(p.x);
        } else {
            self.block_h[p.x].remove(&p.y);
            self.block_w[p.y].remove(&p.x);
        }
        self.block[p] ^= 1;
        if self.block[p] == 1 {
            self.blocks += 1;
        } else {
            self.blocks -= 1;
        }
    }

    fn next_target(&self) -> P { self.targets[self.target_id] }

    fn dist(&self, a: P, b: P) -> i64 {
        // self.dist_bfs(a, b)
        self.dist_manhattan(a, b)
    }

    fn dist_manhattan(&self, a: P, b: P) -> i64 {
        if a == b { return 0; }

        let (minx, maxx) = min_max(a.x, b.x);
        let (miny, maxy) = min_max(a.y, b.y);
        let bx0 = self.block_w[a.y].range(minx..=maxx).count();
        let by0 = self.block_h[b.x].range(miny..=maxy).count();

        let by1 = self.block_h[a.x].range(miny..=maxy).count();
        let bx1 = self.block_w[b.y].range(minx..=maxx).count();
        let block = (bx0+by0).min(by1+bx1);

        (a.manhattan_distance(b) - block) as i64
    }

    fn dist_bfs(&self, a: P, b: P) -> i64 {
        if a == b { return 0; }
        let mut q = deque::new();
        q.push((a, 0));
        let mut vis = FxHashSet::default();
        let mut ret = i64::INF;

        let mut cnt = 0;
        while let Some((v,d)) = q.pop_front() {
            if !vis.insert(v) { continue; }
            cnt += 1;
            chmin!(ret, self.dist_manhattan(v, b) + d);
            // if cnt >= 10 { break; }

            // 右
            let y = self.block_h[v.x].range(v.y+1..).next().cloned().unwrap_or(N-1);
            let nv = P::new(v.x, y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 左
            let y = self.block_h[v.x].range(..v.y).last().cloned().unwrap_or(0);
            let nv = P::new(v.x, y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 下
            let x = self.block_w[v.y].range(v.x+1..).next().cloned().unwrap_or(N-1);
            let nv = P::new(x, v.y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 上
            let x = self.block_w[v.y].range(..v.x).next().cloned().unwrap_or(0);
            let nv = P::new(x, v.y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
        }
        ret
    }

}

impl beam::Context<Op> for Context {
    fn apply(&mut self, op: &Op) {
        match op.t {
            ActionType::Move => {
                self.pos = self.pos.next(op.d);
            },
            ActionType::Slide => {
                self.pos = op.pos;
            },
            ActionType::Alter => {
                self.alter_block(op.pos);
            },
        }
        self.terns += 1;
        self.hash ^= op.hash;
        if op.reached { self.target_id += 1; }

    }

    fn revert(&mut self, op: &Op) {
        match op.t {
            ActionType::Move => {
                self.pos = self.pos.next(op.d.rev());
            },
            ActionType::Slide => {
                self.pos = op.prev_pos;
            },
            ActionType::Alter => {
                self.alter_block(op.pos);
            },
        }
        self.terns -= 1;
        self.hash ^= op.hash;
        if op.reached { self.target_id -= 1; }

    }
}

#[allow(non_camel_case_types)]
type uint=u16;


#[derive(Clone)]
struct State {
    pos: P,
    block: grid_v::GridV<us>,
    block_h: Vec<bset<us>>,
    block_w: Vec<bset<us>>,
    target_id: us,
    targets: Vec<P>,
    terns: us,
    blocks: us,
}

impl State{
    fn new(input:&Input)->State{
        let g = grid_v::GridV::with_default(input.n, input.n, 0);
        let block_h = vec![bset::default(); N];
        let block_w = vec![bset::default(); N];
        Self {
            pos: input.t[0],
            block: g,
            block_h,
            block_w,
            target_id: 1,
            targets: input.t.clone(),
            terns: 0,
            blocks: 0,
        }
    }

    fn apply(&mut self,node:&Node){
        match node.action.t {
            ActionType::Move => {
                self.pos = self.pos.next(node.action.d);
            },
            ActionType::Slide => {
                self.pos = node.action.pos;
            },
            ActionType::Alter => {
                self.alter_block(node.action.pos);
            },
        }
        self.terns += 1;
        if node.reached { self.target_id += 1; }
    }

    fn revert(&mut self,node:&Node){
        match node.action.t {
            ActionType::Move => {
                self.pos = self.pos.next(node.action.d.rev());
            },
            ActionType::Slide => {
                self.pos = node.action.prev_pos;
            },
            ActionType::Alter => {
                self.alter_block(node.action.pos);
            },
        }
        self.terns -= 1;
        if node.reached { self.target_id -= 1; }
    }

    fn next_target(&self) -> P { self.targets[self.target_id] }

    fn alter_block(&mut self, p: P) {
        if self.block[p] == 0 {
            self.block_h[p.x].insert(p.y);
            self.block_w[p.y].insert(p.x);
        } else {
            self.block_h[p.x].remove(&p.y);
            self.block_w[p.y].remove(&p.x);
        }
        self.block[p] ^= 1;
        if self.block[p] == 1 {
            self.blocks += 1;
        } else {
            self.blocks -= 1;
        }
    }
}


#[derive(Clone)]
struct Cand{
    parent:uint,
    eval_score:i64,
    hash:u64,
    action: Op,
    reached: bool,
    is_finished: bool,
}
impl Cand{
    fn raw_score(&self,input:&Input)->i64{
        // ??
        self.eval_score
    }
    
    fn to_node(&self)->Node{
        Node{
            child:!0,
            prev:!0,
            next:!0,
            parent:self.parent,
            score: self.eval_score,
            hash: self.hash,
            action: self.action,
            reached: self.reached,
        }
    }
}


#[derive(Clone,Default)]
struct Node{
    parent:uint,
    child:uint,
    prev:uint,
    next:uint,
    score: i64,
    hash: u64,
    action: Op,
    reached: bool,
}


const MAX_WIDTH:usize=500;
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

    fn restore(&self,mut idx:uint)->Vec<Op>{
        let mut ret=vec![];
        loop{
            let node=&self.nodes[idx as usize];
            if node.parent==!0{
                break;
            }
            ret.push(node.action);
            idx=node.parent;
        }
        
        ret.reverse();
        ret
    }

    fn append_cands(&mut self,input:&Input,idx:usize,cands:&mut Vec<Cand>){
        let node=&self.nodes[idx];
        assert_eq!(node.child,!0);

        self.append_move(input, idx, cands);
        self.append_slide(input, idx, cands);
        self.append_alter(input, idx, cands);
    }

    fn dist(&self, a: P, b: P) -> i64 {
        self.dist_bfs(a, b)
        // self.dist_manhattan(a, b)
    }

    fn dist_manhattan(&self, a: P, b: P) -> i64 {
        if a == b { return 0; }

        let (minx, maxx) = min_max(a.x, b.x);
        let (miny, maxy) = min_max(a.y, b.y);
        let bx0 = self.state.block_w[a.y].range(minx..=maxx).count();
        let by0 = self.state.block_h[b.x].range(miny..=maxy).count();

        let by1 = self.state.block_h[a.x].range(miny..=maxy).count();
        let bx1 = self.state.block_w[b.y].range(minx..=maxx).count();
        let block = (bx0+by0).min(by1+bx1);

        (a.manhattan_distance(b) - block) as i64
    }

    fn dist_bfs(&self, a: P, b: P) -> i64 {
        if a == b { return 0; }
        let mut q = deque::new();
        q.push((a, 0));
        let mut vis = FxHashSet::default();
        let mut ret = i64::INF;

        let mut cnt = 0;
        while let Some((v,d)) = q.pop_front() {
            if !vis.insert(v) { continue; }
            cnt += 1;
            chmin!(ret, self.dist_manhattan(v, b) + d);
            // if cnt >= 10 { break; }

            // 右
            let y = self.state.block_h[v.x].range(v.y+1..).next().cloned().unwrap_or(N-1);
            let nv = P::new(v.x, y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 左
            let y = self.state.block_h[v.x].range(..v.y).last().cloned().unwrap_or(0);
            let nv = P::new(v.x, y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 下
            let x = self.state.block_w[v.y].range(v.x+1..).next().cloned().unwrap_or(N-1);
            let nv = P::new(x, v.y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
            // 上
            let x = self.state.block_w[v.y].range(..v.x).next().cloned().unwrap_or(0);
            let nv = P::new(x, v.y);
            if v.manhattan_distance(nv) >= 2 { q.push((nv, d+1)); }
        }
        ret
    }


    fn append_move(&self,input:&Input,idx:usize,cands:&mut Vec<Cand>) {
        let node=&self.nodes[idx];
        // 隣に移動
        for d in Dir::VAL4 {
            let np = self.state.pos.next(d);
            if !self.state.block.is_in_p(np) || self.state.block[np] == 1 { continue; }

            let dist = self.dist(np, self.state.next_target());
            let reached = np == self.state.next_target();
            let tid = self.state.target_id + if reached { 1 } else { 0 };
            let eval_score =
                (tid - 1).i64() * (T.i64() + 1)
                - (self.state.terns.i64() + dist + 1)
                ;

            let mut hash = node.hash;
            hash ^= input.zob_pos[self.state.pos] ^ input.zob_pos[np];
            // if reached {
            //     if tid - 1 < input.m { hash ^= input.zob_target[tid - 1]; }
            //     if tid < input.m { hash ^= input.zob_target[tid]; }
            // }

            let cand = Cand {
                parent: idx as uint,
                eval_score,
                hash,
                action: Op {t: ActionType::Move, d, prev_pos: self.state.pos, pos: P::INF, reached, hash, },
                reached,
                is_finished: tid == input.m,
            };
            cands.push(cand);
        }
    }

    fn append_slide(&self,input:&Input,idx:usize,cands:&mut Vec<Cand>) {
        let node=&self.nodes[idx];
        // slide
        for d in Dir::VAL4 {
            let mut np = self.state.pos;
            for _ in 0..N {
                let p2 = np.next(d);
                if !self.state.block.is_in_p(p2) || self.state.block[p2] == 1 { break; }
                np = p2;
            }
            if self.state.pos == np { continue; }

            let dist = self.dist(np, self.state.next_target());
            let reached = np == self.state.next_target();
            let tid = self.state.target_id + if reached { 1 } else { 0 };
            let eval_score = (tid - 1).i64() * (T.i64() + 1)
                - (self.state.terns.i64() + dist + 1)
                ;
            let mut hash = node.hash;
            hash ^= input.zob_pos[self.state.pos] ^ input.zob_pos[np];
            // if reached {
            //     if tid - 1 < input.m { hash ^= input.zob_target[tid - 1]; }
            //     if tid < input.m { hash ^= input.zob_target[tid]; }
            // }

            let cand = Cand {
                parent: idx as uint,
                eval_score,
                hash,
                action: Op {t: ActionType::Slide, d, prev_pos: self.state.pos, pos: np, reached, hash,},
                reached,
                is_finished: tid == input.m,
            };
            cands.push(cand);
        }
    }

    fn append_alter(&mut self,input:&Input,idx:usize,cands:&mut Vec<Cand>) {
        let node=&self.nodes[idx];
        for d in Dir::VAL4 {
            let np = self.state.pos.next(d);
            if !self.state.block.is_in_p(np) { continue; }

            self.state.alter_block(np);
            let dist = self.dist(self.state.pos, self.state.next_target());
            let tid = self.state.target_id;
            let eval_score = (tid - 1).i64() * (T.i64() + 1)
                - (self.state.terns.i64() + dist + 1)
                ;
            let mut hash = node.hash;
            hash ^= input.zob_block[np];

            self.state.alter_block(np);
            let cand = Cand {
                parent: idx as uint,
                eval_score,
                hash,
                action: Op {t: ActionType::Alter, d, prev_pos: self.state.pos, pos: np, reached: false, hash, },
                reached: false,
                is_finished: false,
            };
            cands.push(cand);
        }
    }

    fn solve(&mut self,input:&Input)->Vec<Op>{
        use std::cmp::Reverse;
        const MX: us=MAX_WIDTH;

        let mut cands:Vec<Cand>=vec![];
        let mut set=rustc_hash::FxHashSet::default();
        let mut tern = 0;
        for t in 0..T{
            tern = t;
            if t!=0{
                let mut m0=(MX as f64*2.).round() as usize;
                if input.st.elapsed().as_millis() >= 1500 {
                    m0 /= 2;
                }

                if cands.len()>m0{
                    cands.select_nth_unstable_by_key(m0,|a|Reverse(a.eval_score));
                    cands.truncate(m0);
                }
                
                cands.sort_unstable_by_key(|a|Reverse(a.eval_score));
                set.clear();

                let is_finished = cands.iter().any(|c|c.is_finished);
                if is_finished { break; }

                self.update(cands.drain(..).filter(|cand|
                    set.insert(cand.hash)
                ).take(MX));
            }
            
            cands.clear();
            self.enum_cands(input,&mut cands);
            assert!(!cands.is_empty());
        }
    
        let best=cands.into_iter().max_by_key(|a|a.raw_score(input)).unwrap();
        eprintln!("# score={}, tern={}",best.raw_score(input), tern);
        
        let mut ret=self.restore(best.parent);
        ret.push(best.action);
    
        ret
    }
}

pub mod beam {
use fmt::Debug;
use crate::*;

pub use doubly_chained_tree::Context as Context;
pub use doubly_chained_tree::Node as Node;
pub use doubly_chained_tree::NodeId as NodeId;
pub use doubly_chained_tree::NodeValue as NodeValue;

pub trait CandidateAppender<Op: NodeValue, CtxT: Context<Op>> {
    fn append_cands(&self, ctx: &mut CtxT, parent: &Node<Op>, cands: &mut Vec<Cand<Op>>);
}

pub struct Config {
    pub max_width: us,
    pub tern: us,
}

#[derive(Debug)]
pub struct Cand<Op> {
    pub parent: NodeId,
    pub score: i64,
    pub hash: u64,
    pub is_end: bool,
    pub op: Op,
}

pub struct BeamSearch<Op: NodeValue, CtxT: Context<Op>, CandAppenderT: CandidateAppender<Op, CtxT>> {
    cfg: Config,
    ctx: CtxT,
    tree: doubly_chained_tree::DoublyChainedTree<Op>,
    leaf: Vec<NodeId>,
    next_leaf: Vec<NodeId>,
    cand_appender: CandAppenderT,
}

impl<Op, CtxT, CandAppenderT> BeamSearch<Op, CtxT, CandAppenderT>
where 
    Op: NodeValue,
    CtxT: Context<Op>,
    CandAppenderT: CandidateAppender<Op, CtxT> {
    pub fn new(cfg: Config, ctx: CtxT, cand_appender: CandAppenderT) -> Self {
        let max_nodes = cfg.max_width * 5;
        assert!(max_nodes<uint::MAX as usize,"uintのサイズが足りないよ");
        let mut leaf=Vec::with_capacity(cfg.max_width);
        let next_leaf=Vec::with_capacity(cfg.max_width);
        leaf.push(0);

        Self {
            cfg,
            ctx,
            tree: doubly_chained_tree::DoublyChainedTree::new(max_nodes, Op::default()),
            leaf,
            next_leaf,
            cand_appender,
        }
    }

    pub fn solve(&mut self) -> Vec<Op> {
        use std::cmp::Reverse;

        let mut cands: Vec<Cand<Op>> = vec![];
        let mut dup = rustc_hash::FxHashSet::default();
        let mut tern = 0;

        for t in 0..self.cfg.tern {
            tern = t;
            if t != 0 {
                let m0 = self.cfg.max_width * 2;

                if cands.len() > m0 {
                    cands.select_nth_unstable_by_key(m0,|a|Reverse(a.score));
                    cands.truncate(m0);
                } else {
                    cands.sort_unstable_by_key(|a|Reverse(a.score));
                }

                // ターン最小化問題の終了判定 (TODO: 高速化)
                if cands.iter().any(|c|c.is_end) {
                    cands = cands.into_iter().filter(|c|c.is_end).cv();
                    break;
                }

                dup.clear();
                let cands = cands.drain(..)
                    .filter(|cand|dup.insert(cand.hash))
                    .take(self.cfg.max_width);

                self.update(cands);
            }
            
            cands.clear();
            self.enum_cands(&mut cands);
            assert!(!cands.is_empty());
            eprintln!("tern={}/{}, cands={}, free={}", tern, self.cfg.tern, cands.len(), self.tree.free.len());
            // if t == 0 { for c in &cands { eprintln!("{:?}", c); }}
        }
    
        let best = cands.into_iter().max_by_key(|a|a.score).unwrap();
        // eprintln!("# score={}, tern={}",best.score, tern);
        
        let mut ret = self.restore(best.parent);
        ret.push(best.op.clone());
    
        ret
    }

    fn enum_cands(&mut self, cands: &mut Vec<Cand<Op>>) {
        let cand_appender = &self.cand_appender;
        self.tree.walk_leaf(&mut self.ctx, |ctx, parent| {
            cand_appender.append_cands(ctx, parent, cands);
        });
    }

    fn update(&mut self, cands: impl Iterator<Item=Cand<Op>>){
        self.next_leaf.clear();
        for cand in cands { self.tree.add_node(cand.parent, cand.op.clone()); }

        for &id in &self.leaf {
            if !self.tree.nodes[id as usize].has_child() {
                self.tree.remove_node(id);
            }
        }

        std::mem::swap(&mut self.leaf,&mut self.next_leaf);
    }

    fn restore(&self, mut idx: NodeId)-> Vec<Op>{
        let mut ret=vec![];
        loop {
            let node = &self.tree.nodes[idx as usize];
            if node.is_root() { break; }
            ret.push(node.value.clone());
            idx = node.parent;
        }
        
        ret.reverse();
        ret
    }



}

}


// #CAP(fumin::modint)
pub mod fumin {
pub mod grid {
#![allow(dead_code)]
use std::ops::{Index, IndexMut};
use itertools::iproduct;

use crate::common::*;
use super::pt::{Pt, Dir};


pub struct Grid<T>(pub Vec<Vec<T>>);

impl<T: Clone> Grid<T> {
    pub fn new(h: us, w: us, v: T) -> Self { Self::from(&vec![vec![v; w]; h]) }
}
impl<T> Grid<T> {
    pub fn h(&self) -> us { self.0.len() }
    pub fn w(&self) -> us { self.0[0].len() }
    pub fn is_in_p(&self, p: Pt<us>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t(&self, t: (us,us)) -> bool { t.0 < self.h() && t.1 < self.w() }
}
impl<T: Clone+Eq> Grid<T> {
    pub fn position(&self, t: &T) -> Option<Pt<us>> {
        iproduct!(0..self.h(), 0..self.w()).into_iter().map(|(i,j)|Pt::<us>::new(i,j)).filter(|&p|self[p]==*t).next()
    }
}
impl<T: Clone> From<&Vec<Vec<T>>> for Grid<T> {
    fn from(v: &Vec<Vec<T>>) -> Self { Self(v.to_vec()) }
}
impl<T, N: IntoT<us>> Index<Pt<N>> for Grid<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: IntoT<us>> IndexMut<Pt<N>> for Grid<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T, N: IntoT<us>> Index<(N,N)> for Grid<T> {
    type Output = T;
    fn index(&self, p: (N,N)) -> &Self::Output { &self.0[p.0.us()][p.1.us()] }
}
impl<T, N: IntoT<us>> IndexMut<(N,N)> for Grid<T> {
    fn index_mut(&mut self, p: (N,N)) -> &mut Self::Output { &mut self.0[p.0.us()][p.1.us()] }
}
impl<T, N: IntoT<us>> Index<N> for Grid<T> {
    type Output = Vec<T>;
    fn index(&self, p: N) -> &Self::Output { &self.0[p.us()] }
}
impl<T, N: IntoT<us>> IndexMut<N> for Grid<T> {
    fn index_mut(&mut self, p: N) -> &mut Self::Output { &mut self.0[p.us()] }
}
impl Grid<char> {
    pub fn bfs(&self, s: Pt<us>) -> Grid<us> {
        let mut que = deque::new();
        let mut m = Grid::new(self.h(), self.w(), us::INF);
        que.push_back(s);
        m[s] = 0;
        while let Some(v) = que.pop_front() {
            for d in Dir::VAL4 {
                let nv = v.next(d);
                if self.is_in_p(nv) && self[nv]!='#' && m[nv]==us::INF {
                    m[nv] = m[v]+1;
                    que.push_back(nv);
                }
            }
        }
        m
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
        // aから見てbがどちらの方向にあるか
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
            .map(|(&a,&b)|Dir::dir(a,b))
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
            if x == y { break; }
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

impl Default for Dir {
    fn default() -> Self { Dir::R }
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
pub mod doubly_chained_tree {
use itertools::Itertools;

pub type NodeId = u16;
const INF: NodeId = !0;

pub trait NodeValue : std::fmt::Debug + Clone + Default {}

#[derive(Debug, Clone, Default)]
pub struct Node<T: NodeValue> {
    pub id: NodeId,
    pub parent: NodeId,
    child: NodeId,
    prev: NodeId,
    next: NodeId,
    pub value: T,
}

impl<T: NodeValue> Node<T> {
    pub fn is_root(&self) -> bool { self.parent == INF }
    pub fn has_child(&self) -> bool { self.child != INF }
}

pub trait Context<T: NodeValue> {
    fn apply(&mut self, value: &T);
    fn revert(&mut self, value: &T);
}


pub struct DoublyChainedTree<T: NodeValue> {
    pub nodes:Vec<Node<T>>,
    pub free:Vec<NodeId>,
}

impl<T: NodeValue> DoublyChainedTree<T> {
    pub fn new(max_nodes: usize, root: T) -> Self {
        let mut nodes = vec![Node::default(); max_nodes];
        nodes[0] = Node {id: 0, parent: INF, child: INF, prev: INF, next: INF, value: root};
        let free=(1..nodes.len() as NodeId).rev().collect_vec();
        Self {
            nodes,
            free,
        }
    }

    pub fn reset(&mut self, root: Node<T>) {
        self.nodes[0] = root;
        self.free.clear();
        self.free.extend((1..self.nodes.len() as NodeId).rev());
    }

    pub fn add_node(&mut self, parent: NodeId, value: T) -> NodeId {
        // Nodeのイメージ
        // (追加前)
        // 1
        // v
        // 2 > 3
        // (4を追加)
        // 1
        // v
        // 4 > 2 > 3

        // 新しいNodeを親の子の兄弟として追加する
        let next = self.nodes[parent as usize].child;
        let new = if let Some(n) = self.free.pop() {
            self.nodes[n as usize] = Node{id: n, parent, next, child: INF, prev: INF, value};
            n
        } else {
            let n = self.nodes.len() as NodeId;
            assert!(n!=0,"Not enough size for NodeId");
            self.nodes.push(Node{id:n, parent, next, child: INF, prev: INF, value});
            n
        };

        // 兄弟が既にいる場合、その兄弟のprevに新しいNodeを追加
        if next != INF { self.nodes[next as usize].prev = new; }

        // 親の子として新しいNodeを追加
        self.nodes[parent as usize].child = new;

        new
    }

    pub fn remove_node(&mut self, mut idx: NodeId) {
        loop{
            self.free.push(idx);
            let Node{prev,next,parent,..} = self.nodes[idx as usize];
            assert_ne!(parent, INF,"全てのノードを消そうとしています");

            // 削除対象Nodeが一人っ子の場合、親Nodeを残す意味がないため削除する
            if prev & next == INF {
                idx = parent;
                continue;
            }

            // 削除対象Nodeのnextを付け替え
            if prev != INF {
                self.nodes[prev as usize].next = next;
            } else{
                self.nodes[parent as usize].child = next;
            }

            // 削除対象Nodeのprevを付け替え
            if next != INF {
                self.nodes[next as usize].prev = prev;
            }
            
            break;
        }
    }

    pub fn walk_leaf<C: Context<T>>(&self, ctx: &mut C, mut walker: impl FnMut(&mut C, &Node<T>)) {
        let mut cur_node = 0;
        loop {
            let Node{next,child,..} = self.nodes[cur_node];
            if next==INF || child==INF { break; }
            cur_node = child as usize;
            ctx.apply(&self.nodes[cur_node].value);
        }

        let root = cur_node;
        loop {
            let child = self.nodes[cur_node].child;
            if child == INF {
                walker(ctx, &self.nodes[cur_node]);

                loop {
                    if cur_node == root { return; }
                    let node = &self.nodes[cur_node];
                    ctx.revert(&node.value);
                    // 兄弟に移動
                    if node.next != INF {
                        cur_node = node.next as usize;
                        ctx.apply(&self.nodes[cur_node].value);
                        break;
                    }
                    // 親に移動
                    cur_node = node.parent as usize;
                }
            } else {
                // 子に移動
                cur_node = child as usize;
                ctx.apply(&self.nodes[cur_node].value);
            }
        }

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
