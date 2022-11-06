use std::*;
use proconio::{input, fastout};
use fumin::*;

#[fastout]
fn main() {
}

#[allow(dead_code, unused_macros)]
pub mod fumin {
use std::*;

pub type Us  = usize;
pub type Is  = isize;
pub type Us1 = proconio::marker::Usize1;
pub type Is1 = proconio::marker::Isize1;

#[macro_export] macro_rules! def_caster {
    ($($t:ty; $f:ident),*) => { $(#[macro_export] macro_rules! $f { ($n:expr) => { (($n) as $t) } })* };
}
def_caster!(usize;us, isize;is, i32;i32, i64;i64);

pub fn recurfn<P, R>(p: P, f: &dyn Fn(P, &dyn Fn(P) -> R) -> R) -> R { f(p, &|p: P| recurfn(p, &f)) }
pub fn chmax<T: PartialOrd+Copy>(target: &mut T, value: &T) -> bool { chif(target, value, cmp::Ordering::Greater) }
pub fn chmin<T: PartialOrd+Copy>(target: &mut T, value: &T) -> bool { chif(target, value, cmp::Ordering::Less) }
fn chif<T: PartialOrd+Copy>(target: &mut T, value: &T, cond: std::cmp::Ordering) -> bool {
    if value.partial_cmp(target) == Some(cond) { *target = value.clone(); true } else { false }
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

#[cfg(all(test, feature="template"))]
#[path="./main_test.rs"]
mod tests;