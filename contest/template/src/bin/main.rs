use std::*;
use proconio::{input, fastout};
use fumin::*;

#[fastout]
fn main() {
}

#[allow(dead_code, unused_macros)]
pub mod fumin {
use std::{*, ops::*};

pub type Us  = usize;
pub type Is  = isize;
pub type Us1 = proconio::marker::Usize1;
pub type Is1 = proconio::marker::Isize1;
pub type Chars = proconio::marker::Chars;
pub type Bytes = proconio::marker::Bytes;

pub trait PartialPrimNum: Copy+PartialOrd<Self>+Add<Output=Self>+Sub<Output=Self>+Mul<Output=Self>+Div<Output=Self>+Rem<Output=Self> {
    const ZERO: Self;
    const ONE:  Self;
    fn us(self) -> usize;
    fn is(self) -> isize;
    fn from_is(n: isize) -> Self;
}
pub trait PartialIPrimNum: PartialPrimNum+Neg<Output=Self> {
    const MONE: Self;
}
pub trait PrimInt: PartialPrimNum+Ord {
    fn trailing_zeros(self) -> u32;
    fn pow(self, exp: u32) -> Self;
}
pub trait IPrimInt : PrimInt+PartialIPrimNum {}

#[macro_export] macro_rules! impl_partial_prim_num {
    ($($t:ty),*) => {$(
        impl PartialPrimNum for $t {
            const ZERO: Self = 0 as Self;
            const ONE:  Self = 1 as Self;
            fn us(self) -> usize { self as usize }
            fn is(self) -> isize { self as isize }
            fn from_is(n: isize) -> Self { n as Self }
        }
    )*}
}
#[macro_export] macro_rules! impl_partial_iprim_num {
    ($($t:ty),*) => {$(
        impl PartialIPrimNum for $t {
            const MONE: Self = -1 as Self;
        }
    )*}
}

#[macro_export] macro_rules! impl_prim_int {
    ($($t:ty),*) => {$(
        impl PrimInt for $t {
            fn trailing_zeros(self) -> u32 { self.trailing_zeros() }
            fn pow(self, exp: u32) -> Self { self.pow(exp) }
        }
    )*}
}
#[macro_export] macro_rules! impl_iprim_int {
    ($($t:ty),*) => {$(
        impl IPrimInt for $t {}
    )*}
}

impl_partial_prim_num! (isize, i32, i64, f32, f64, usize, u32, u64);
impl_partial_iprim_num!(isize, i32, i64, f32, f64);
impl_prim_int! (isize, i32, i64, usize, u32, u64);
impl_iprim_int!(isize, i32, i64);

#[derive(Debug,PartialEq,Copy,Clone)]
pub struct Pt<N: PartialIPrimNum> { pub x: N, pub y: N }
impl<N: PartialIPrimNum> ops::Add<Pt<N>> for Pt<N> {
    type Output = Pt<N>;
    fn add(self, rhs: Pt<N>) -> Self::Output { Pt{x: self.x + rhs.x, y: self.y + rhs.y} }
}
impl<N: PartialIPrimNum> Pt<N> {
    pub fn dir4() -> [Pt<N>; 4] {
        [Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0), Pt::from_is(-1,0)]
    }
    pub fn dir8() -> [Pt<N>; 8] {
        [Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0), Pt::from_is(-1,0),
         Pt::from_is(1,1), Pt::from_is(1,-1), Pt::from_is(-1,1), Pt::from_is(-1,1) ]
    }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn from_is(x: isize, y: isize) -> Pt<N> { Self::of(N::from_is(x), N::from_is(y)) }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
}

pub fn recurfn<P, R>(p: P, f: &dyn Fn(P, &dyn Fn(P) -> R) -> R) -> R { f(p, &|p: P| recurfn(p, &f)) }
pub fn chmax<N: PrimInt>(target: &mut N, value: &N) -> bool { chif(target, value, cmp::Ordering::Greater) }
pub fn chmin<N: PrimInt>(target: &mut N, value: &N) -> bool { chif(target, value, cmp::Ordering::Less) }
pub fn abs_diff<N: PrimInt>(n1: N, n2: N) -> N { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
fn chif<N: PrimInt>(target: &mut N, value: &N, cond: std::cmp::Ordering) -> bool {
    if value.partial_cmp(target) == Some(cond) { *target = value.clone(); true } else { false }
}

pub struct CumSum<N> { pub s: Vec<N> }
impl<N: Default+PrimInt> CumSum<N> {
    pub fn new(v: &Vec<N>) -> Self {
        let mut cs = CumSum{ s: Vec::new() };
        cs.s.resize(v.len() + 1, Default::default());
        for i in 0..v.len() { cs.s[i+1] = cs.s[i] + v[i]; }
        cs
    }
    pub fn sum(&self, l: usize, r: usize) -> N { self.s[r] - self.s[l] }
}

pub struct Grid<T> { pub raw: Vec<Vec<T>> }
impl<T: Clone> Grid<T> {
    pub fn from(v: &Vec<Vec<T>>) -> Grid<T> { Grid{raw: v.to_vec()} }
    pub fn new(h: Us, w: Us, v: T) -> Grid<T> { Grid{raw: vec![vec![v; w]; h]} }
    pub fn inp<N:PartialIPrimNum>(&self, p: Pt<N>) -> bool { self.inij(p.x, p.y) }
    pub fn inij<N:PartialIPrimNum>(&self, i: N, j: N) -> bool { 0<=i.is() && i.is()<self.raw.len().is() && 0<=j.is() && j.is()<self.raw[i.us()].len().is() }
}
impl<T, N: PartialIPrimNum> Index<Pt<N>> for Grid<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: PartialIPrimNum> IndexMut<Pt<N>> for Grid<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T, N: PartialIPrimNum> Index<(N,N)> for Grid<T> {
    type Output = T;
    fn index(&self, p: (N,N)) -> &Self::Output { &self.raw[p.0.us()][p.1.us()] }
}
impl<T, N: PartialIPrimNum> IndexMut<(N,N)> for Grid<T> {
    fn index_mut(&mut self, p: (N,N)) -> &mut Self::Output { &mut self.raw[p.0.us()][p.1.us()] }
}

trait Joiner { fn join(self, sep: &str) -> String; }
impl<It: Iterator<Item=String>> Joiner for It { fn join(self, sep: &str) -> String { self.collect::<Vec<_>>().join(sep) } }

pub trait Fmtx { fn fmtx(&self) -> String; }
macro_rules! fmtx_primitive { ($($t:ty),*) => { $(impl Fmtx for $t { fn fmtx(&self) -> String { self.to_string() }})* } }

fmtx_primitive! {
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128,
    usize, isize, f32, f64, char, &str, String, bool
}

pub struct ByLine<'a, T> { pub v: &'a Vec<T> }
impl<'a, T: Fmtx> ByLine<'a, T> { pub fn from(v: &'a Vec<T>) -> ByLine<'a, T> { ByLine{v} }}

impl<'a, T: Fmtx> Fmtx for ByLine<'a, T> {
    fn fmtx(&self) -> String { self.v.iter().map(|e| e.fmtx()).join("\n") }
}
impl<T: Fmtx> Fmtx for Vec<T> {
    fn fmtx(&self) -> String { self.iter().map(|e| e.fmtx()).join(" ") }
}
impl<K: fmt::Display, V: Fmtx> Fmtx for collections::HashMap<K, V> {
    fn fmtx(&self) -> String { self.iter().map(|(k,v)| format!("{}:{}", k, v.fmtx())).join(" ") }
}

#[macro_export] macro_rules! fmtx {
    ($a:expr, $($b:expr),*)       => {{ format!("{} {}", fmtx!(($a)), fmtx!($($b),*)) }};
    ($a:expr)                     => {{ ($a).fmtx() }};

    ($a:expr, $($b:expr),*;debug) => {{ format!("{} {}", fmtx!(($a);debug), fmtx!($($b),*;debug)) }};
    ($a:expr;debug)               => {{ format!("{:?}", ($a)) }};

    ($a:expr, $($b:expr),*;line)  => {{ format!("{}\n{}", fmtx!(($a);line), fmtx!($($b),*;line)) }};
    ($a:expr;line)                => {{ ($a).fmtx() }};

    ($a:expr;byline)              => {{ format!("{}", ByLine{v:&($a)}.fmtx()) }};
}

#[macro_export] macro_rules! out {
    ($($a:expr),*)        => { println!("{}", fmtx!($($a),*)); };
    ($($a:expr),*;debug)  => { println!("{}", fmtx!($($a),*;debug)); };
    ($($a:expr),*;line)   => { println!("{}", fmtx!($($a),*;line)); };
    ($a:expr;byline)      => { println!("{}", fmtx!($a;byline)); };
}

macro_rules! scream {
    ($yes:ident, $no:ident) => {
        #[macro_export] macro_rules! $yes {
            ($b:expr) => { out!(if $b { stringify!($yes) } else { stringify!($no) }); };
            ()        => { out!(stringify!($yes)); };
        }
    };
}

macro_rules! yesno {
    ($yes:ident, $no:ident) => { scream!($yes, $no); scream!($no, $yes); };
}

yesno!(yes, no);
yesno!(Yes, No);
yesno!(YES, NO);

}

#[cfg(all(test, feature="template"))]
#[path="../tests/main_test.rs"]
mod tests;