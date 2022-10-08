use std::*;
use proconio::{input, fastout};
use fumin::*;

#[fastout]
fn main() {
}

#[allow(dead_code, unused_macros)]
pub mod fumin {
use std::*;

#[macro_export] macro_rules! vvec {
    // vvec![0; m, n] => vec![vec![0; m]; n]
    ($x:expr; $s:expr) => { vec![$x; $s] };
    ($x:expr; $s0:expr; $($s:expr);+) => { vvec![vec![$x; $s0]; $($s);+ ] };
}

pub fn chif<T: PartialOrd>(value: T, target: &mut T, cond: std::cmp::Ordering) -> bool {
    if value.partial_cmp(target) == Some(cond) { *target = value; true } else { false }
}
#[macro_export] macro_rules! chmax {
    ($target:expr, $value:expr) => { chif($value, &mut $target, cmp::Ordering::Greater) };
}
#[macro_export] macro_rules! chmin {
    ($target:expr, $value:expr) => { chif($value, &mut $target, cmp::Ordering::Less) };
}

pub struct CumSum<N> { pub s: Vec<N> }
impl<N: Default+ops::Add<Output=N>+ops::Sub<Output=N>+Copy> CumSum<N> {
    pub fn new(v: &Vec<N>) -> Self {
        let mut cs = CumSum{ s: Vec::new() };
        cs.s.resize(v.len() + 1, Default::default());
        for i in 0..v.len() { cs.s[i+1] = cs.s[i] + v[i]; }
        cs
    }
    pub fn sum(&self, l: usize, r: usize) -> N { self.s[r] - self.s[l] }
}

trait Joiner { fn join(self, sep: &str) -> String; }
impl<It: Iterator<Item=String>> Joiner for It { fn join(self, sep: &str) -> String { self.collect::<Vec<_>>().join(sep) } }

pub trait Fmtx<T=Self> { fn fmtx(&self) -> String; }
macro_rules! fmtx_primitive { ($($t:ty),*) => { $(impl Fmtx for $t { fn fmtx(&self) -> String { self.to_string() }})* } }

fmtx_primitive! {
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128,
    usize, isize, f32, f64, char, &str, String, bool
}

pub struct ByLine<'a, T> { v: &'a Vec<T> }
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

#[macro_export] macro_rules! out {
    ($($a:expr),*) => {
        let mut v = Vec::<String>::new();
        $(v.push($a.fmtx());)*
        println!("{}", v.fmtx());
    };
}

macro_rules! scream {
    ($yes:ident, $no:ident) => {
        #[macro_export]
        macro_rules! $yes {
            ($b:expr) => { out!(if $b { stringify!($yes) } else { stringify!($no) }); };
            () => { out!(stringify!($yes)); };
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

// #[cfg(test)] #[path = "./main_test.rs"] mod fumintests;