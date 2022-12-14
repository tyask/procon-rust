#![allow(dead_code, unused_imports, unused_macros, non_snake_case, non_camel_case_types)]
use std::{*, ops::*, collections::*, iter::Sum};

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

pub trait FromT<T> { fn from_t(t: T) -> Self; }
pub trait IntoT<T> { fn into_t(self) -> T; }
pub trait Unit {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const TEN: Self;
}

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
{}
 
pub trait ExPrimInt: SimplePrimInt
        + Rem<Output=Self>
        + RemAssign
        + Unit
{}

pub trait ToUs   { fn us(self) -> us; }
pub trait ToIs   { fn is(self) -> is; }
pub trait ToF64  { fn f64(self) -> f64; }
pub trait ToU8   { fn u8(self) -> u8; }
pub trait ToChar { fn char(self) -> char; }

#[macro_export] macro_rules! impl_prim_num {
    ($($t:ty),*) => {$(
        impl SimplePrimInt for $t { }
        impl ExPrimInt     for $t { }
        impl Unit for $t {
            const ZERO: Self =  0 as Self;
            const ONE : Self =  1 as Self;
            const TWO : Self =  2 as Self;
            const TEN : Self = 10 as Self;
        }
        impl FromT<us>   for $t { fn from_t(n: us) -> Self { n as $t } }
        impl FromT<is>   for $t { fn from_t(n: is) -> Self { n as $t } }
        impl IntoT<us>   for $t { fn into_t(self)  -> us   { self as us } }
        impl IntoT<is>   for $t { fn into_t(self)  -> is   { self as is } }
        impl IntoT<f64>  for $t { fn into_t(self)  -> f64  { self as f64 } }
        impl IntoT<u8>   for $t { fn into_t(self)  -> u8   { self as u8 } }
        impl IntoT<char> for $t { fn into_t(self)  -> char { (self as u8) as char } }
        impl IntoT<u64>  for $t { fn into_t(self)  -> u64  { self as u64 } }
        impl ToUs        for $t { fn us(self)      -> us   { self as us } }
        impl ToIs        for $t { fn is(self)      -> is   { self as is } }
        impl ToF64       for $t { fn f64(self)     -> f64  { self as f64 } }
        impl ToU8        for $t { fn u8(self)      -> u8   { self as u8 } }
        impl ToChar      for $t { fn char(self)    -> char { (self as u8) as char } }
    )*}
}

impl_prim_num! {isize, i8, i32, i64, usize, u8, u32, u64, f32, f64}

// Utilities
#[macro_export] macro_rules! or { ($cond:expr;$a:expr,$b:expr) => { if $cond { $a } else { $b } }; }
struct Rec<'s, P, R=()> { f: &'s dyn Fn(&Self, P) -> R }
impl<'s, P, R> Rec<'s, P, R> {
    pub fn new(f: &'s dyn Fn(&Self, P) -> R) -> Self { Self { f: f } }
    pub fn call(&self, p: P) -> R { (self.f)(self, p) }
}
pub fn chmax<N: Clone+PartialOrd>(value: &N, target: &mut N) -> bool { chif(value, target, cmp::Ordering::Greater) }
pub fn chmin<N: Clone+PartialOrd>(value: &N, target: &mut N) -> bool { chif(value, target, cmp::Ordering::Less) }
fn chif<N:Clone+PartialOrd>(value: &N, target: &mut N, cond: std::cmp::Ordering) -> bool {
    if value.partial_cmp(target) == Some(cond) { *target = value.clone(); true } else { false }
}

pub fn abs_diff<N: SimplePrimInt>    (n1: N, n2: N)       -> N { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
pub fn gcd<N:    ExPrimInt>          (mut a: N, mut b: N) -> N { while b > N::ZERO { let c = b; b = a % b; a = c; } a }
pub fn lcm<N:    ExPrimInt>          (a: N, b: N)         -> N { if a==N::ZERO || b==N::ZERO { N::ZERO } else { a / gcd(a,b) * b }}
pub fn floor<N:  SimplePrimInt>      (a: N, b: N)         -> N { a / b }
pub fn ceil<N:   SimplePrimInt+Unit> (a: N, b: N)         -> N { (a + b - N::ONE) / b }
pub fn modulo<N: ExPrimInt>          (n: N, m: N)         -> N { let r = n % m; or!(r < N::ZERO; r + m, r) }
pub fn powmod<N: ExPrimInt+Unit+BitAnd<Output=N>+Shr<Output=N>>     (mut n: N, mut k: N, m: N) -> N {
    // n^k mod m
    let one = N::ONE;
    let mut a = one;
    while k > N::ZERO {
        if k & one == one { a *= n; a %= m; }
        n %= m; n *= n; n %= m;
        k = k >> one;
    }
    a
}
pub fn sumae<N: SimplePrimInt+Unit>(n: N, a: N, e: N) -> N { n * (a + e) / N::TWO }
pub fn sumad<N: SimplePrimInt+Unit>(n: N, a: N, d: N) -> N { n * (N::TWO * a + (n - N::ONE) * d) / N::TWO }
pub fn ndigits<N: SimplePrimInt+Unit>(mut n: N) -> usize { let mut d = 0; while n > N::ZERO { d+=1; n/=N::TEN; } d }
pub fn asc<T:Ord>(a: &T, b: &T)  -> cmp::Ordering { a.cmp(b) }
pub fn desc<T:Ord>(a: &T, b: &T) -> cmp::Ordering { b.cmp(a) }

pub trait IterTrait : Iterator where Self::Item: hash::Hash+Eq {
    fn counts<N: SimplePrimInt+Unit>(&mut self) -> map<Self::Item, N> {
        self.fold(map::<Self::Item, N>::new(), |mut m: map<Self::Item,N>, x| { *m.or_def_mut(x) += N::ONE; m })
    }
}

impl<T: ?Sized> IterTrait for T where T: Iterator, Self::Item: hash::Hash+Eq { }

// Vec
pub trait VecFill<T>  { fn fill(&mut self, t: T); }
pub trait VecOrDef<T> { fn or_def(&self, i: us) -> T; }
pub trait VecOr<T>    { fn or<'a>(&'a self, i: us, v: &'a  T) -> &'a T; }

impl<T:Clone+Copy> VecFill<T>          for Vec<T> { fn fill(&mut self, t: T) { self.iter_mut().for_each(|x| *x = t); } }
impl<T:Clone+Copy+Default> VecOrDef<T> for Vec<T> { fn or_def(&self, i: us) -> T { self.get(i).cloned().unwrap_or_default() } }
impl<T:Clone+Copy> VecOr<T>            for Vec<T> { fn or<'a>(&'a self, i: us, v: &'a T) -> &'a T  { self.get(i).unwrap_or(v) } }

// Map
pub trait MapOrDef<K,V> { fn or_def(&self, k: &K) -> V; }
pub trait MapOrDefMut<K,V> { fn or_def_mut(&mut self, k: K) -> &mut V; }
pub trait MapOr<K,V> { fn or<'a>(&'a self, k: &K, v: &'a  V) -> &'a V; }

impl<K:Eq+hash::Hash, V:Default+Clone> MapOrDef<K, V> for map<K, V> {
    fn or_def(&self, k: &K) -> V { self.get(&k).cloned().unwrap_or_default() }
}
impl<K:Eq+hash::Hash, V:Default> MapOrDefMut<K, V> for map<K, V> {
    fn or_def_mut(&mut self, k: K) -> &mut V { self.entry(k).or_default() }
}
impl<K:Eq+hash::Hash, V> MapOr<K, V> for map<K, V> {
    fn or<'a>(&'a self, k: &K, v: &'a V) -> &'a V  { self.get(&k).unwrap_or(v) }
}

impl<K:Ord, V:Default+Clone> MapOrDef<K, V> for bmap<K, V> {
    fn or_def(&self, k: &K) -> V { self.get(&k).cloned().unwrap_or_default() }
}
impl<K:Ord, V:Default> MapOrDefMut<K, V> for bmap<K, V> {
    fn or_def_mut(&mut self, k: K) -> &mut V { self.entry(k).or_default() }
}
impl<K:Ord, V> MapOr<K, V> for bmap<K, V> {
    fn or<'a>(&'a self, k: &K, v: &'a V) -> &'a V  { self.get(&k).unwrap_or(v) }
}

pub trait BSetTrait<T> {
    fn lower_bound(&self, t: &T) -> Option<&T>;
    fn upper_bound(&self, t: &T) -> Option<&T>;
}
impl<T:Ord> BSetTrait<T> for bset<T> {
    fn lower_bound(&self, t: &T) -> Option<&T> { self.range(t..).next() }
    fn upper_bound(&self, t: &T) -> Option<&T> { self.range((Bound::Excluded(t), Bound::Unbounded)).next() }
}

// Graph
pub fn digraph(n: us, uv: &Vec<(usize, usize)>) -> Vec<Vec<us>> {
    let mut g = vec![vec![]; n]; uv.iter().for_each(|&(u,v)|g[u].push(v)); g
}
pub fn undigraph(n: us, uv: &Vec<(usize, usize)>) -> Vec<Vec<us>> {
    let mut g = vec![vec![]; n]; uv.iter().for_each(|&(u,v)|{g[u].push(v); g[v].push(u);}); g
}

// Pt
#[derive(Debug,Copy,Clone,PartialEq,Eq,Hash,PartialOrd,Ord,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N: SimplePrimInt> Pt<N> {
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn on(self, h: Range<N>, w: Range<N>) -> bool { h.contains(&self.x) && w.contains(&self.y) }
}
impl<N: SimplePrimInt+FromT<is>> Pt<N> {
    pub fn dir4() -> Vec<Pt<N>> {
        vec![Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0), Pt::from_is(-1,0)]
    }
    pub fn dir8() -> Vec<Pt<N>> {
        vec![
            Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0),  Pt::from_is(-1,0),
            Pt::from_is(1,1), Pt::from_is(1,-1), Pt::from_is(-1,1), Pt::from_is(-1,-1)
            ]
    }
    pub fn from_is(x: is, y: is) -> Pt<N> { Self::of(N::from_t(x), N::from_t(y)) }
}
impl<N: SimplePrimInt+FromT<is>+ToF64> Pt<N> {
    pub fn norm(self)  -> f64 { self.norm2().f64().sqrt() }
}
impl Pt<f64> {
    pub fn rot(self, r: f64) -> Pt<f64> {
        let (x, y) = (self.x, self.y);
        Self::of(r.cos()*x-r.sin()*y, r.sin()*x+r.cos()*y) // ??????????????????r?????????(r???radian)
    }
}
impl<N: SimplePrimInt+fmt::Display> fmt::Display  for Pt<N> { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{} {}", self.x, self.y) } }
impl<N: SimplePrimInt+fmt::Display> Fmt           for Pt<N> { fn fmt(&self) -> String { format!("{} {}", self.x, self.y) } }
impl<N: SimplePrimInt> AddAssign<Pt<N>> for Pt<N> { fn add_assign(&mut self, rhs: Pt<N>) { self.x = self.x + rhs.x; self.y = self.y + rhs.y; } }
impl<N: SimplePrimInt> SubAssign<Pt<N>> for Pt<N> { fn sub_assign(&mut self, rhs: Pt<N>) { self.x = self.x - rhs.x; self.y = self.y - rhs.y; } }
impl<N: SimplePrimInt> MulAssign<N>     for Pt<N> { fn mul_assign(&mut self, rhs: N) { self.x *= rhs; self.y *= rhs; } }
impl<N: SimplePrimInt> DivAssign<N>     for Pt<N> { fn div_assign(&mut self, rhs: N) { self.x /= rhs; self.y /= rhs; } }
impl<N: SimplePrimInt> Add<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: Pt<N>) -> Self::Output { self += rhs; self } }
impl<N: SimplePrimInt> Sub<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: Pt<N>) -> Self::Output { self -= rhs; self } }
impl<N: SimplePrimInt> Mul<N>           for Pt<N> { type Output = Pt<N>; fn mul(mut self, rhs: N) -> Self::Output { self *= rhs; self } }
impl<N: SimplePrimInt> Div<N>           for Pt<N> { type Output = Pt<N>; fn div(mut self, rhs: N) -> Self::Output { self /= rhs; self } }
impl<N: SimplePrimInt+Default> Sum      for Pt<N> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::default(), |a, b| a + b) } }

impl<N: SimplePrimInt+FromT<is>+proconio::source::Readable<Output=N>> proconio::source::Readable for Pt<N> {
    type Output = Pt<N>;
    fn read<R: io::BufRead, S: proconio::source::Source<R>>(source: &mut S) -> Self::Output {
        Pt::of(N::read(source), N::read(source))
    }
}


// CumSum
pub struct CumSum<N> { pub s: Vec<N> }
impl<N: SimplePrimInt> CumSum<N> {
    pub fn new(v: &Vec<N>) -> Self {
        let mut cs = CumSum{ s: Vec::new() };
        cs.s.resize(v.len() + 1, Default::default());
        for i in 0..v.len() { cs.s[i+1] = cs.s[i] + v[i]; }
        cs
    }
    pub fn sum(&self, l: usize, r: usize) -> N { self.s[r] - self.s[l] }
}

// Grid
pub struct Grid<T> { pub raw: Vec<Vec<T>> }
impl<T: Clone> Grid<T> {
    pub fn from(v: &Vec<Vec<T>>) -> Grid<T> { Grid{raw: v.to_vec()} }
    pub fn new(h: us, w: us, v: T) -> Grid<T> { Grid{raw: vec![vec![v; w]; h]} }
    pub fn inp<N:SimplePrimInt+ToUs+ToIs>(&self, p: Pt<N>)    -> bool { self.inij(p.x, p.y) }
    pub fn inij<N:SimplePrimInt+ToUs+ToIs>(&self, i: N, j: N) -> bool { 0<=i.is() && i.is()<self.raw.len().is() && 0<=j.is() && j.is()<self.raw[i.us()].len().is() }
    pub fn int<N:SimplePrimInt+ToUs+ToIs>(&self, t: (N, N))   -> bool { self.inij(t.0, t.1) }
}
impl<T, N: SimplePrimInt+ToUs> Index<Pt<N>> for Grid<T> {
    type Output = T;
    fn index(&self, p: Pt<N>) -> &Self::Output { &self[p.tuple()] }
}
impl<T, N: SimplePrimInt+ToUs> IndexMut<Pt<N>> for Grid<T> {
    fn index_mut(&mut self, p: Pt<N>) -> &mut Self::Output { &mut self[p.tuple()] }
}
impl<T, N: SimplePrimInt+ToUs> Index<(N,N)> for Grid<T> {
    type Output = T;
    fn index(&self, p: (N,N)) -> &Self::Output { &self.raw[p.0.us()][p.1.us()] }
}
impl<T, N: SimplePrimInt+ToUs> IndexMut<(N,N)> for Grid<T> {
    fn index_mut(&mut self, p: (N,N)) -> &mut Self::Output { &mut self.raw[p.0.us()][p.1.us()] }
}

// io

// ?????????????????????????????????????????????input?????????
// let src = from_stdin();
// input! {from src, n: usize}
pub fn from_stdin() -> proconio::source::line::LineSource<io::BufReader<io::Stdin>> {
    proconio::source::line::LineSource::new(io::BufReader::new(io::stdin()))
}

trait Joiner { fn join(self, sep: &str) -> String; }
impl<It: Iterator<Item=String>> Joiner for It { fn join(self, sep: &str) -> String { self.collect::<Vec<_>>().join(sep) } }

pub trait Fmt { fn fmt(&self) -> String; }
macro_rules! fmt_primitive { ($($t:ty),*) => { $(impl Fmt for $t { fn fmt(&self) -> String { self.to_string() }})* } }

fmt_primitive! {
    u8, u16, u32, u64, u128, i8, i16, i32, i64, i128,
    usize, isize, f32, f64, char, &str, String, bool
}

impl<T: Fmt> Fmt for Vec<T>      { fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") } }
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
    ($($a:expr),*) => { eprintln!("{}", fmt!(@debug $($a),*)); }
}
#[macro_export]#[cfg(not(feature="local"))] macro_rules! debug {
    ($($a:expr),*) => { }
}

pub fn yes(b: bool) -> &'static str { if b { "yes" } else { "no" } }
pub fn Yes(b: bool) -> &'static str { if b { "Yes" } else { "No" } }
pub fn YES(b: bool) -> &'static str { if b { "YES" } else { "NO" } }
pub fn no(b: bool) -> &'static str { yes(!b) }
pub fn No(b: bool) -> &'static str { Yes(!b) }
pub fn NO(b: bool) -> &'static str { YES(!b) }

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {

    use crate::common::*;

    #[test]
    fn test_fmtx() {
        assert_eq!(fmt!(2),           "2");
        assert_eq!(fmt!(2, 3),        "2 3");
        assert_eq!(fmt!(2.123),       "2.123");
        assert_eq!(fmt!(vec![1,2,3]), "1 2 3");

        assert_eq!(fmt!(@line 2, 3),        "2\n3");

        assert_eq!(fmt!(@byline vec![1,2,3]),          "1\n2\n3");
        assert_eq!(fmt!(@byline vec!["ab","cd","ef"]), "ab\ncd\nef");

        assert_eq!(fmt!(@debug vec![1,2,3]), "[1, 2, 3]");
    }


    #[test]
    fn test_chmax_chmin() {
        {
            let mut m = 0;
            let mut do_chmax = |v, exp_updated, exp_val| {
                assert_eq!(chmax(v, &mut m), exp_updated);
                assert_eq!(m, exp_val);
            };
            do_chmax(&1, true,  1);
            do_chmax(&1, false, 1);
            do_chmax(&0, false, 1);
        }

        {
            let mut m = 1;
            let mut do_chmin = |v, exp_updated, exp_val| {
                assert_eq!(chmin(v, &mut m), exp_updated);
                assert_eq!(m, exp_val);
            };

            do_chmin(&0, true,  0);
            do_chmin(&0, false, 0);
            do_chmin(&1, false, 0);
        }
    }
}