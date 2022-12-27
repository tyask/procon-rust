#[allow(unused_imports)]
use std::{*, collections::*, ops::*, cmp::*};
use proconio::{input, fastout};
use fumin::*;

#[fastout]
fn main() {
}

#[allow(dead_code, unused_macros, non_snake_case, non_camel_case_types)]
pub mod fumin {
use std::{*, ops::*, collections::*, fmt::*, iter::Sum};

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

// PrimNum
pub trait PartialPrimNum:
        Copy
        +PartialOrd<Self>
        +Add<Output=Self>
        +Sub<Output=Self>
        +Mul<Output=Self>
        +Div<Output=Self>
        +Rem<Output=Self>
        +MulAssign
        +DivAssign
        +RemAssign
        {
    const ZERO: Self;
    const ONE:  Self;
    fn us(self)   -> us;
    fn is(self)   -> is;
    fn f64(self)  -> f64;
    fn u8(self)   -> u8;
    fn char(self) -> char;
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
            fn us(self)   -> us   { self as usize }
            fn is(self)   -> is   { self as isize }
            fn f64(self)  -> f64  { self as f64   }
            fn u8(self)   -> u8   { self as u8    }
            fn char(self) -> char { self.u8() as char }
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

// Utilities
#[macro_export] macro_rules! or { ($cond:expr;$a:expr,$b:expr) => { if $cond { $a } else { $b } }; }
pub fn recurfn<P, R>(p: P, f: &dyn Fn(P, &dyn Fn(P) -> R) -> R) -> R { f(p, &|p: P| recurfn(p, &f)) }
pub fn chmax<N: Clone+PartialOrd>(value: &N, target: &mut N) -> bool { chif(value, target, cmp::Ordering::Greater) }
pub fn chmin<N: Clone+PartialOrd>(value: &N, target: &mut N) -> bool { chif(value, target, cmp::Ordering::Less) }
fn chif<N:Clone+PartialOrd>(value: &N, target: &mut N, cond: std::cmp::Ordering) -> bool {
    if value.partial_cmp(target) == Some(cond) { *target = value.clone(); true } else { false }
}

pub fn abs_diff<N: PrimInt>(n1: N, n2: N)       -> N { if n1 >= n2 { n1 - n2 } else { n2 - n1 } }
pub fn gcd<N:PrimInt>      (mut a: N, mut b: N) -> N { while b > N::ZERO { let c = b; b = a % b; a = c; } a }
pub fn lcm<N:PrimInt>      (a: N, b: N)         -> N { if a==N::ZERO || b==N::ZERO { N::ZERO } else { a / gcd(a,b) * b }}
pub fn floor<N:PrimInt>    (a: N, b: N)         -> N { a / b }
pub fn ceil<N:PrimInt>     (a: N, b: N)         -> N { (a + b - N::ONE) / b }
pub fn modulo<N:PrimInt>   (n: N, m: N)         -> N { let r = n % m; or!(r < N::ZERO; r + m, r) }
pub fn powmod<N:PrimInt+BitAnd<Output=N>+Shr<Output=N>+ShrAssign>(mut n: N, mut k: N, m: N) -> N {
    // n^k mod m
    let one = N::ONE;
    let mut a = one;
    while k > N::ZERO {
        if k & one == one { a *= n; a %= m; }
        n %= m; n *= n; n %= m;
        k >>= one;
    }
    a
}
pub fn sumae<N:PrimInt>(n: N, a: N, e: N) -> N { n * (a + e) / N::from_is(2) }
pub fn sumad<N:PrimInt>(n: N, a: N, d: N) -> N { n * (N::from_is(2) * a + (n - N::ONE) * d) / N::from_is(2) }
pub fn ndigits<N:PrimInt>(mut n: N) -> usize { let mut d = 0; while n > N::ZERO { d+=1; n/=N::from_is(10); } d }
pub fn asc<T:Ord>(a: &T, b: &T)  -> cmp::Ordering { a.cmp(b) }
pub fn desc<T:Ord>(a: &T, b: &T) -> cmp::Ordering { b.cmp(a) }

pub trait IterTrait : Iterator where Self::Item: hash::Hash+Eq {
    fn counts(&mut self) -> map<Self::Item, us> {
        self.fold(map::<Self::Item, us>::new(), |mut m: map<Self::Item, us>, x| { *m.or_def_mut(x) += 1; m })
    }
}

impl<T: ?Sized> IterTrait for T where T: Iterator, Self::Item: hash::Hash+Eq { }

pub trait VecTrait<T> {
    fn fill(&mut self, t: T);
}
impl<T:Clone+Copy> VecTrait<T> for Vec<T> {
    fn fill(&mut self, t: T) { self.iter_mut().for_each(|x| *x = t); }
}

pub trait MapTrait<K,V> {
    fn or_def(&self, k: &K)        -> V;
    fn or(&self, k: &K, v: V)      -> V;
    fn or_def_mut(&mut self, k: K) -> &mut V;
}

impl<K:Eq+hash::Hash, V:Default+Clone> MapTrait<K, V> for map<K, V> {
    fn or_def(&self, k: &K)        -> V      { self.get(&k).cloned().unwrap_or_default() }
    fn or(&self, k: &K, v: V)      -> V      { self.get(&k).cloned().unwrap_or(v) }
    fn or_def_mut(&mut self, k: K) -> &mut V { self.entry(k).or_default() }
}
impl<K:Eq+Ord, V:Default+Clone> MapTrait<K, V> for bmap<K, V> {
    fn or_def(&self, k: &K)        -> V      { self.get(&k).cloned().unwrap_or_default() }
    fn or(&self, k: &K, v: V)      -> V      { self.get(&k).cloned().unwrap_or(v) }
    fn or_def_mut(&mut self, k: K) -> &mut V { self.entry(k).or_default() }
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
#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord,Default)]
pub struct Pt<N: PartialIPrimNum> { pub x: N, pub y: N }

impl<N: PartialIPrimNum> Pt<N> {
    pub fn dir4() -> Vec<Pt<N>> {
        vec![Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0), Pt::from_is(-1,0)]
    }
    pub fn dir8() -> Vec<Pt<N>> {
        vec![
            Pt::from_is(0,1), Pt::from_is(0,-1), Pt::from_is(1,0),  Pt::from_is(-1,0),
            Pt::from_is(1,1), Pt::from_is(1,-1), Pt::from_is(-1,1), Pt::from_is(-1,-1)
            ]
    }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn from_is(x: isize, y: isize) -> Pt<N> { Self::of(N::from_is(x), N::from_is(y)) }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn norm(self)  -> f64 { self.norm2().f64().sqrt() }
}
impl Pt<f64> {
    pub fn rot(self, r: f64) -> Pt<f64> {
        let (x, y) = (self.x, self.y);
        Self::of(r.cos()*x-r.sin()*y, r.sin()*x+r.cos()*y) // 反時計回りにr度回転(rはradian)
    }
}
impl<N: PartialIPrimNum+Display> Display  for Pt<N> { fn fmt(&self, f: &mut Formatter<'_>) -> Result { write!(f, "{} {}", self.x, self.y) } }
impl<N: PartialIPrimNum+Display> Fmt      for Pt<N> { fn fmt(&self) -> String { format!("{} {}", self.x, self.y) } }
impl<N: PartialIPrimNum> AddAssign<Pt<N>> for Pt<N> { fn add_assign(&mut self, rhs: Pt<N>) { self.x = self.x + rhs.x; self.y = self.y + rhs.y; } }
impl<N: PartialIPrimNum> SubAssign<Pt<N>> for Pt<N> { fn sub_assign(&mut self, rhs: Pt<N>) { self.x = self.x - rhs.x; self.y = self.y - rhs.y; } }
impl<N: PartialIPrimNum> MulAssign<N>     for Pt<N> { fn mul_assign(&mut self, rhs: N) { self.x *= rhs; self.y *= rhs; } }
impl<N: PartialIPrimNum> DivAssign<N>     for Pt<N> { fn div_assign(&mut self, rhs: N) { self.x /= rhs; self.y /= rhs; } }
impl<N: PartialIPrimNum> Add<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn add(mut self, rhs: Pt<N>) -> Self::Output { self += rhs; self } }
impl<N: PartialIPrimNum> Sub<Pt<N>>       for Pt<N> { type Output = Pt<N>; fn sub(mut self, rhs: Pt<N>) -> Self::Output { self -= rhs; self } }
impl<N: PartialIPrimNum> Mul<N>           for Pt<N> { type Output = Pt<N>; fn mul(mut self, rhs: N) -> Self::Output { self *= rhs; self } }
impl<N: PartialIPrimNum> Div<N>           for Pt<N> { type Output = Pt<N>; fn div(mut self, rhs: N) -> Self::Output { self /= rhs; self } }
impl<N: PartialIPrimNum+Default> Sum      for Pt<N> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::default(), |a, b| a + b) } }

impl<N: PartialIPrimNum + proconio::source::Readable<Output=N>> proconio::source::Readable for Pt<N> {
    type Output = Pt<N>;
    fn read<R: io::BufRead, S: proconio::source::Source<R>>(source: &mut S) -> Self::Output {
        Pt::of(N::read(source), N::read(source))
    }
}


// CumSum
pub struct CumSum<N> { pub s: Vec<N> }
impl<N: Default+Add<Output=N>+Sub<Output=N>+Copy+Clone> CumSum<N> {
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
    pub fn inp<N:PartialIPrimNum>(&self, p: Pt<N>)    -> bool { self.inij(p.x, p.y) }
    pub fn inij<N:PartialIPrimNum>(&self, i: N, j: N) -> bool { 0<=i.is() && i.is()<self.raw.len().is() && 0<=j.is() && j.is()<self.raw[i.us()].len().is() }
    pub fn int<N:PartialIPrimNum>(&self, t: (N, N))   -> bool { self.inij(t.0, t.1) }
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

// io

/// インタラクティブ問題ではこれをinputに渡す
/// ```
/// let src = from_stdin();
/// input! {from src, n: usize}
/// ```
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

impl<T: Fmt> Fmt for Vec<T> {
    fn fmt(&self) -> String { self.iter().map(|e| e.fmt()).join(" ") }
}
impl<K: Display, V: Fmt> Fmt for HashMap<K, V> {
    fn fmt(&self) -> String { self.iter().map(|(k,v)| format!("{}:{}", k, v.fmt())).join(" ") }
}

#[macro_export] macro_rules! fmt {
    ($a:expr, $($b:expr),*)       => {{ format!("{} {}", fmt!(($a)), fmt!($($b),*)) }};
    ($a:expr)                     => {{ ($a).fmt() }};

    (@debug $a:expr, $($b:expr),*) => {{ format!("{} {}", fmt!(@debug ($a)), fmt!(@debug $($b),*)) }};
    (@debug $a:expr)               => {{ format!("{:?}", ($a)) }};

    (@line $a:expr, $($b:expr),*) => {{ format!("{}\n{}", fmt!(@line $a), fmt!(@line $($b),*)) }};
    (@line $a:expr)               => {{ ($a).fmt() }};

    (@byline $a:expr) => {{ use itertools::Itertools; ($a).iter().map(|e| e.fmt()).join("\n") }};
    (@gird   $a:expr) => {{ use itertools::Itertools; ($a).iter().map(|v| v.iter().collect::<Str>()).join("\n") }};
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

}

#[cfg(all(test, feature="template"))]
#[path="./main_test.rs"]
mod tests;