#![allow(unused_imports)]
use std::{*, collections::*, ops::*, cmp::*, iter::*, io::stdin};
use itertools::Itertools;
use ::num::{Zero, One};
use proconio::{input, fastout};
use common::*;

fn main() {
    solve();
}

#[fastout]
fn solve() {
    // CONTEST(abc200-a)
}

// #CAP(fumin::modint)
pub mod fumin {
}

pub mod common {
#![allow(dead_code, unused_imports, unused_macros, non_snake_case, non_camel_case_types)]
use std::{*, ops::*, collections::*, iter::Sum};
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
    fn wraping_add(self, a: Self) -> Self;
}
impl Wrapping for us  { fn wraping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for is  { fn wraping_add(self, a: Self) -> Self { self.wrapping_add(a) } }
impl Wrapping for i64 { fn wraping_add(self, a: Self) -> Self { self.wrapping_add(a) } }

// Utilities
#[macro_export] macro_rules! or    { ($cond:expr;$a:expr,$b:expr) => { if $cond { $a } else { $b } }; }
#[macro_export] macro_rules! chmax { ($a:expr,$b:expr) => { { let v = $b; if $a < v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! chmin { ($a:expr,$b:expr) => { { let v = $b; if $a > v { $a = v; true } else { false } } } }
#[macro_export] macro_rules! add   { ($a:expr,$b:expr) => { { let v = $b; $a += v; } } }
#[macro_export] macro_rules! sub   { ($a:expr,$b:expr) => { { let v = $b; $a -= v; } } }
#[macro_export] macro_rules! mul   { ($a:expr,$b:expr) => { { let v = $b; $a *= v; } } }
#[macro_export] macro_rules! div   { ($a:expr,$b:expr) => { { let v = $b; $a /= v; } } }
#[macro_export] macro_rules! rem   { ($a:expr,$b:expr) => { { let v = $b; $a %= v; } } }

pub fn abs_diff(n1: us, n2: us) -> us { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
pub fn floor<N: SimplePrimInt>(a: N, b: N) -> N { a / b }
pub fn ceil<N: SimplePrimInt>(a: N, b: N) -> N { (a + b - N::one()) / b }
pub fn asc <T:Ord>(a: &T, b: &T) -> cmp::Ordering { a.cmp(b) }
pub fn desc<T:Ord>(a: &T, b: &T) -> cmp::Ordering { b.cmp(a) }

pub trait IterTrait : Iterator {
    fn counts<N: SimplePrimInt+FromT<us>>(&mut self) -> map<Self::Item, N> where Self::Item: hash::Hash+Eq+Clone {
        self.fold(map::<_,_>::new(), |mut m, x| { *m.or_def_mut(&x) += N::from_t(1); m })
    }
    fn grouping_to_bmap<'a, K:Ord+Clone, V>(&'a mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> bmap<K, Vec<V>> {
        self.fold(bmap::<_,_>::new(), |mut m, x| { m.or_def_mut(&get_key(&x)).push(get_val(&x)); m })
    }
    fn grouping_to_map<K:Eq+hash::Hash+Clone, V>(&mut self, get_key: impl Fn(&Self::Item)->K, get_val: impl Fn(&Self::Item)->V) -> map<K, Vec<V>> {
        self.fold(map::<_,_>::new(), |mut m, x| { m.or_def_mut(&get_key(&x)).push(get_val(&x)); m })
    }
    fn cv(&mut self) -> Vec<Self::Item> { self.collect_vec() }
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

impl<T> IterTrait     for T where T: Iterator { }
impl<T, U: IntoT<char>> CharIterTrait<U> for T where T: Iterator<Item=U> { }
impl<T> HashIterTrait for T where T: Iterator, Self::Item: Eq+hash::Hash { }
impl<T> OrdIterTrait  for T where T: Iterator, Self::Item: Ord { }

// Vec
pub trait VecFill<T>  { fn fill(&mut self, t: T); }
pub trait VecOrDef<T> { fn or_def(&self, i: us) -> T; }
pub trait VecOr<T>    { fn or<'a>(&'a self, i: us, v: &'a  T) -> &'a T; }
pub trait VecMax<T>    { fn vmax(&self) -> T; }
pub trait VecMin<T>    { fn vmin(&self) -> T; }
pub trait VecSum<T>    { fn sum(&self) -> T; }

impl<T:Clone>         VecFill<T>  for [T] { fn fill(&mut self, t: T) { self.iter_mut().for_each(|x| *x = t.clone()); } }
impl<T:Clone+Default> VecOrDef<T> for [T] { fn or_def(&self, i: us) -> T { self.get(i).cloned().unwrap_or_default() } }
impl<T:Clone+Copy>    VecOr<T>    for [T] { fn or<'a>(&'a self, i: us, v: &'a T) -> &'a T  { self.get(i).unwrap_or(v) } }
impl<T:Clone+Ord>     VecMax<T>   for [T] { fn vmax(&self) -> T  { self.iter().cloned().max().unwrap() } }
impl<T:Clone+Ord>     VecMin<T>   for [T] { fn vmin(&self) -> T  { self.iter().cloned().min().unwrap() } }
impl<T:Clone+Sum<T>>  VecSum<T>   for [T] { fn sum(&self)  -> T  { self.iter().cloned().sum::<T>() } }

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

#[macro_export] macro_rules! fmt {
    ($a:expr, $($b:expr),*)       => {{ format!("{} {}", fmt!(($a)), fmt!($($b),*)) }};
    ($a:expr)                     => {{ ($a).fmt() }};

    (@debug $a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(@debug ($a)), fmt!(@debug $($b),*)) }};
    (@debug $a:expr)               => {{ format!("{:?}", ($a)) }};

    (@line $a:expr, $($b:expr),*) => {{ format!("{}\n{}", fmt!(@line $a), fmt!(@line $($b),*)) }};
    (@line $a:expr)               => {{ ($a).fmt() }};

    (@byline $a:expr) => {{ use itertools::Itertools; ($a).iter().map(|e| e.fmt()).join("\n") }};
    (@grid   $a:expr) => {{ use itertools::Itertools; ($a).iter().map(|v| v.iter().collect::<Str>()).join("\n") }};
}

#[macro_export]#[cfg(feature="local")] macro_rules! debug {
    ($($a:expr),*)    => { eprintln!("{}", fmt!(@debug  $($a),*)); };
    (@byline $a:expr) => { eprintln!("{}", fmt!(@byline $a)); };
    (@grid   $a:expr) => { eprintln!("{}", fmt!(@grid   $a)); };
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug {
    ($($a:expr),*)    => { };
    (@byline $a:expr) => { };
    (@grid   $a:expr) => { };
}

pub fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }
pub fn Yes(b: bool) -> &'static str { if b { "Yes" } else { "No" } }
pub fn YES(b: bool) -> &'static str { if b { "YES" } else { "NO" } }
pub fn no(b: bool) -> &'static str { yes(!b) }
pub fn No(b: bool) -> &'static str { Yes(!b) }
pub fn NO(b: bool) -> &'static str { YES(!b) }


}

