#![allow(unused_imports)]
use std::{cmp::*, collections::*, iter::*, mem::swap, ops::*, rc::Rc, time::Instant, *};
use itertools::{iproduct, Itertools};
use proconio::{input, fastout};
use common::*;
use fumin::{grid_v::GridV, pt::{Dir, Pt}, *};
use rand::{seq::IteratorRandom, Rng, RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;

fn main() {
    solve();
}

type P = Pt<us>;
type G = GridV<Cell>;
const LIMIT_MS: u128 = 1800;

impl G {
    fn swap_a(&mut self, x: P, y: P) {
        let t = self[x].a;
        self[x].a = self[y].a;
        self[y].a = t;
    }

    fn eval_all(&self) -> i64 {
        let mut s = 0;
        for (i,j) in iproduct!(0..self.h,0..self.w) {
            let p = P::new(i,j);
            for &d in &self[p].dirs {
                s += self[p].eval(&self[p.next(d)]);
            }
        }
        s
    }
}

impl P {
    fn next_d(self, d: Option<Dir>) -> P {
        if d.is_some() { self.next(d.unwrap()) } else { self }
    }
}

struct Input {
    st: Instant,
    t: us,
    n: us,
    g: G,
}

impl Input {
    fn new() -> Self {
        let st = Instant::now();
        input! {t:us,n:us,v:[chars;n],h:[chars;n-1],a:[[i64;n];n]}

        let mut g = G::new(n, n);
        for i in 0..n { for j in 0..n {
            let p = P::new(i,j);
            g[p].a = a[p.x][p.y];
            for dir in Dir::VAL4 {
                let d = dir.p();
                let np = p.wrapping_add(d);
                if np.x >= n || np.y >= n { continue; }
                let mn = P::new(min(p.x,np.x), min(p.y,np.y));
                if d.x != 0 && mn.x < n-1 && h[mn.x][p.y]=='1' { continue; }
                if d.y != 0 && mn.y < n-1 && v[p.x][mn.y]=='1' { continue; }
                g[p].dirs.push(dir);
            }
        }}

        Self {
            st,
            t,
            n,
            g,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Node<T> {
    children: Vec<Rc<Node<T>>>,
    value: T,
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Self {
            children: vec![],
            value,
        }
    }

    fn push(&mut self, t: T) {
        self.children.push(Rc::new(Node::new(t)));
    }
}

#[derive(Debug, Clone, Default)]
struct Action {
    s: us, // 交換するなら1, しないなら0
    d: Option<Dir>, // 高橋君の移動先
    e: Option<Dir>, // 青木君の移動先
    score: i64, // 変化したスコア
}

impl Action {
    fn format(&self) -> String {
        format!("{} {} {}",
            self.s,
            self.d.map(|d|d.c()).unwrap_or('.'),
            self.e.map(|d|d.c()).unwrap_or('.')
        )
    }
}

#[derive(Debug, Clone, Default)]
struct Out {
    ini_x: P,
    ini_y: P,
    actions: Vec<Action>,
    score: i64,
}

impl Out {
    #[fastout]
    fn print(&self) {
        println!("{} {}", self.ini_x, self.ini_y);
        for a in &self.actions { println!("{}", a.format()); }
    }
}

struct State<'a> {
    input: &'a Input,
    g: G,
    score: us,
    out: Out,
    x: P, // 高橋
    y: P, // 青木
    rng: Pcg64Mcg,
    hurry: bool,
    ncand: Vec<(i64,i64,Option<Dir>,Option<Dir>,P,P,Option<Dir>,Option<Dir>,P,P,Option<Dir>,Option<Dir>)>,
}

impl<'a> State<'a> {
    fn new(input: &'a Input) -> Self {
        Self {
            input,
            g: input.g.clone(),
            score: 0,
            out: Out::default(),
            x: P::new(0,0),
            y: P::new(0,0),
            // rng: Pcg64Mcg::seed_from_u64(7),
            rng: Pcg64Mcg::from_entropy(),
            hurry: false,
            ncand: vec![],
        }
    }

    fn initialize(&mut self) {
        let n = self.input.n;
        let rng = &mut self.rng;
        self.out.ini_x = P::new(rng.gen_range(0..n), rng.gen_range(0..n));
        self.out.ini_y = P::new(rng.gen_range(0..n), rng.gen_range(0..n));
        self.x = self.out.ini_x;
        self.y = self.out.ini_y;
    }

    fn execute(&mut self) {
        let input = self.input;
        for i in 0..4 * input.n * input.n {
            if i & 0xFF == 0 {
                let elapsed = input.st.elapsed().as_millis();
                if elapsed >= LIMIT_MS { break; }
                if elapsed >= 1500 { self.hurry = true; }
            }
            self.execute_tern(i);
        }
    }

    fn execute_tern(&mut self, tern: us) {
        let input = self.input;
        let mut act = Action::default();
        let (x, y) = (self.x, self.y);
        let e = self.eval_swap(x, y);
        act.score = e;
        // debug!(tern, e);
        if e < 0 {
            act.s = 1;
            self.g.swap_a(x, y);
        }

        let mut c = if input.n <= 30 && !self.hurry {
            self.find_best3(x, y)
        } else {
            self.find_best2(x, y)
        };

        if c.0 >= 0 {
            c.1 = self.g[x].choose(&mut self.rng);
            c.2 = self.g[y].choose(&mut self.rng);
            self.ncand.clear();
        }

        act.d = c.1;
        act.e = c.2;
        self.x = x.next_d(act.d);
        self.y = y.next_d(act.e);
        self.out.actions.push(act);
    }

    fn find_best2(&self, x: P, y: P) -> (i64, Option<Dir>, Option<Dir>) {
        let mut best = (i64::INF, None, None);
        for &xd in &self.g[x].dirs2() { for &yd in &self.g[y].dirs2() {
            let (nx, ny) = (x.next_d(xd), y.next_d(yd));
            let s = self.eval_swap(nx,ny).min(0);
            for &xd2 in &self.g[nx].dirs2() { for &yd2 in &self.g[ny].dirs2() {
                let (nx2, ny2) = (nx.next_d(xd2), ny.next_d(yd2));
                let s = s + self.eval_swap(nx2,ny2).min(0);
                if chmin!(best.0, s) {
                    best.1 = xd;
                    best.2 = yd;
                }
            }}
        }}
        best
    }

    fn find_best3(&mut self, x: P, y: P) -> (i64, Option<Dir>, Option<Dir>) {
        let mut cand = vec![];
        if self.ncand.is_empty() {
            for &xd in &self.g[x].dirs2() { for &yd in &self.g[y].dirs2() {
                let (nx, ny) = (x.next_d(xd), y.next_d(yd));
                let s = self.eval_swap(nx,ny).min(0);
                for &xd2 in &self.g[nx].dirs2() { for &yd2 in &self.g[ny].dirs2() {
                    let (nx2, ny2) = (nx.next_d(xd2), ny.next_d(yd2));
                    let s2 = s + self.eval_swap(nx2,ny2).min(0);
                    cand.push((s,s2,xd,yd,nx2,ny2,xd2,yd2));
                }}
            }}
        } else {
            for &(s2,s3,xd,yd,nx2,ny2,xd2,yd2,nx3,ny3,xd3,yd3) in &self.ncand {
                cand.push((s2,s3,xd2,yd2,nx3,ny3,xd3,yd3));
            }
        }

        self.ncand.clear();
        let mut best = (i64::INF, None, None);
        for (s,s2,xd,yd,nx2,ny2,xd2,yd2) in cand {
            for &xd3 in &self.g[nx2].dirs2() { for &yd3 in &self.g[ny2].dirs2() {
                let (nx3, ny3) = (nx2.next_d(xd3), ny2.next_d(yd3));
                let t3 = self.eval_swap(nx3,ny3).min(0);
                let s3 = s2 + t3;
                self.ncand.push((s2,s2-s+t3,xd,yd,nx2,ny2,xd2,yd2,nx3,ny3,xd3,yd3));
                if chmin!(best.0, s3) {
                    best.1 = xd;
                    best.2 = yd;
                }
            }}
        }

        if best.0 != i64::INF {
            self.ncand.retain(|t|(t.2,t.3)==(best.1,best.2));
        } else {
            self.ncand.clear();
        }
        self.ncand.clear();
        best
    }

    fn eval_swap(&self, x: P, y: P) -> i64 {
        let cur = self.eval_p(x, self.g[x].a) + self.eval_p(y, self.g[y].a);
        let nxt = self.eval_p(x, self.g[y].a) + self.eval_p(y, self.g[x].a);
        nxt - cur
    }

    fn eval_p(&self, p: P, a: i64) -> i64 {
        let mut s = 0;
        for &d in &self.g[p].dirs {
            s += self.g[p.next(d)].eval_a(a);
        }
        s
    }

}

#[derive(Clone, Debug, Default)]
struct Cell {
    dirs: Vec<Dir>,
    a: i64,
}

impl Cell {
    fn eval(&self, c: &Cell) -> i64 { (self.a - c.a) * (self.a - c.a) }
    fn eval_a(&self, a: i64) -> i64 { (self.a - a) * (self.a - a) }
    fn choose(&self, rng: &mut impl RngCore) -> Option<Dir> {
        let x = rng.gen_range(0..=self.dirs.len());
        self.dirs.get(x).cloned()
    }

    fn dirs2(&self) -> Vec<Option<Dir>> {
        self.dirs.iter().map(|&d|Some(d)).chain([None]).cv()
    }
}

// CONTEST(abcXXX-a)
// #[fastout]
fn solve() {
    let input = Input::new();
    let mut best = Out::default();
    best.score = i64::INF;
    // let mut g = G::new(input.n, input.n);
    let mut cnt = 0;
    loop {
        let elapsed = input.st.elapsed().as_millis();
        if elapsed >= LIMIT_MS { break; }
        cnt += 1;
        let mut st = State::new(&input);
        st.initialize();
        st.execute();
        let score = st.g.eval_all();
        if best.score > score {
            best = st.out.clone();
            best.score = score;
            // g = st.g;
        }
    }

    best.print();
    eprintln!("# T={},N={},CNT={},ACTIONS={}", input.t, input.n, cnt, best.actions.len());
    // for i in 0..input.n {
    //     eprintln!("{}", fmt!(g[i].iter().map(|c|c.a).cv()));
    // }
}

// #CAP(fumin::modint)
pub mod fumin {
pub mod grid_v {
#![allow(dead_code)]
use std::{ops::{Index, IndexMut}, cmp::Reverse};

use crate::{common::*, chmin};
use super::pt::{Pt, Dir};


#[derive(Clone)]
pub struct GridV<T> {
    pub g: Vec<T>,
    pub h: us,
    pub w: us,
}

impl <T> GridV<T>
where
    T: Clone + Default {
    pub fn new(h: us, w: us) -> Self {
        Self { g: vec![T::default(); h * w], h, w, }
    }
    pub fn with_default(h: us, w: us, v: T) -> Self {
        Self { g: vec![v; h * w], h, w, }
    }
    pub fn is_in_p<N: IntoT<us>>(&self, p: Pt<N>) -> bool { self.is_in_t(p.tuple()) }
    pub fn is_in_t<N: IntoT<us>>(&self, t: (N, N)) -> bool { t.0.into_t() < self.h && t.1.into_t() < self.w }
}

impl<T, N: IntoT<us>> Index<N> for GridV<T> {
    type Output = [T];
    fn index(&self, i: N) -> &Self::Output {
        let idx = i.into_t() * self.h;
        &self.g[idx..idx+self.w]
    }
}
impl<T, N: IntoT<us>> IndexMut<N> for GridV<T> {
    fn index_mut(&mut self, i: N) -> &mut Self::Output {
        let idx = i.into_t() * self.h;
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
use num_traits::Signed;
use rand::Rng;

use crate::{common::*, enrich_enum, count};


// Pt
#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Hash,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N> Pt<N> {
    pub fn new(x: impl IntoT<N>, y: impl IntoT<N>) -> Pt<N> { Pt{x:x.into_t(), y:y.into_t()} }
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
    pub fn prev(self, d: Dir) -> Self { self.wrapping_sub(d.p()) }
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
impl<N: MulAssign<N>+Copy> MulAssign<N>     for Pt<N> { fn mul_assign(&mut self, rhs: N) { self.x *= rhs; self.y *= rhs; } }
impl<N: DivAssign<N>+Copy> DivAssign<N>     for Pt<N> { fn div_assign(&mut self, rhs: N) { self.x /= rhs; self.y /= rhs; } }
impl<N: RemAssign<N>+Copy> RemAssign<N>     for Pt<N> { fn rem_assign(&mut self, rhs: N) { self.x %= rhs; self.y %= rhs; } }
impl<N: AddAssign<N>+Copy> Add<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: Pt<N>) -> Self::Output { self += rhs; self } }
impl<N: SubAssign<N>+Copy> Sub<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: Pt<N>) -> Self::Output { self -= rhs; self } }
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
pub fn minmax<T: Ord+Copy>(a: T, b: T) -> (T, T) { (cmp::min(a,b), cmp::max(a,b)) }
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
        self.fold(bmap::<_,_>::new(), |mut m, x| { m.or_def_mut(&get_key(&x)).push(get_val(&x)); m })
    }
    fn grouping_to_map<K:Eq+hash::Hash+Clone, V>(&mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> map<K, Vec<V>> {
        self.fold(map::<_,_>::new(), |mut m, x| { m.or_def_mut(&get_key(&x)).push(get_val(&x)); m })
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
            *cnt.or_def_mut(&e) += C::one();
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
pub trait VecFill<T> { fn fill(&mut self, t: T); }
impl<T:Clone> VecFill<T> for [T] { fn fill(&mut self, t: T) { self.iter_mut().for_each(|x| *x = t.clone()); } }

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

// Map
pub trait MapOrDef<K,V> { fn or_def(&self, k: &K) -> V; }
pub trait MapOrDefMut<K,V> { fn or_def_mut(&mut self, k: &K) -> &mut V; }
pub trait MapOr<K,V> { fn or<'a>(&'a self, k: &K, v: &'a  V) -> &'a V; }

impl<K:Eq+hash::Hash, V:Default+Clone> MapOrDef<K, V> for map<K, V> {
    fn or_def(&self, k: &K) -> V { self.get(&k).cloned().unwrap_or_default() }
}
impl<K:Eq+hash::Hash+Clone, V:Default> MapOrDefMut<K, V> for map<K, V> {
    fn or_def_mut(&mut self, k: &K) -> &mut V { self.entry(k.clone()).or_default() }
}
impl<K:Eq+hash::Hash, V> MapOr<K, V> for map<K, V> {
    fn or<'a>(&'a self, k: &K, v: &'a V) -> &'a V  { self.get(&k).unwrap_or(v) }
}

impl<K:Ord, V:Default+Clone> MapOrDef<K, V> for bmap<K, V> {
    fn or_def(&self, k: &K) -> V { self.get(&k).cloned().unwrap_or_default() }
}
impl<K:Ord+Clone, V:Default> MapOrDefMut<K, V> for bmap<K, V> {
    fn or_def_mut(&mut self, k: &K) -> &mut V { self.entry(k.clone()).or_default() }
}
impl<K:Ord, V> MapOr<K, V> for bmap<K, V> {
    fn or<'a>(&'a self, k: &K, v: &'a V) -> &'a V  { self.get(&k).unwrap_or(v) }
}
pub trait BMapTrait<K,V> {
    fn lower_bound(&self, k: &K) -> Option<(&K, &V)>;
    fn upper_bound(&self, k: &K) -> Option<(&K, &V)>;
}
impl<K:Ord, V> BMapTrait<K, V> for bmap<K, V> {
    fn lower_bound(&self, k: &K) -> Option<(&K, &V)> { self.range(k..).next() }
    fn upper_bound(&self, k: &K) -> Option<(&K, &V)> { self.range((Bound::Excluded(k), Bound::Unbounded)).next() }
}

pub trait BSetTrait<T> {
    fn lower_bound(&self, t: &T) -> Option<&T>;
    fn upper_bound(&self, t: &T) -> Option<&T>;
}
impl<T:Ord> BSetTrait<T> for bset<T> {
    fn lower_bound(&self, t: &T) -> Option<&T> { self.range(t..).next() }
    fn upper_bound(&self, t: &T) -> Option<&T> { self.range((Bound::Excluded(t), Bound::Unbounded)).next() }
}

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
