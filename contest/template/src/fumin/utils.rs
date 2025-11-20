#![allow(dead_code)]
use std::{*, ops::*};
use num_traits::{Zero, One};

use crate::{common::*, add};

// 素数判定
pub fn is_price(n: us) -> bool {
    if n <= 1 { return false; }
    let mut i = 2;
    while i * i <= n { if n % i == 0 { return false; } i += 1; }
    true
}

// 約数を列挙する
pub fn divisors(n: us) -> Vec<us> {
    let mut a = Vec::new();
    let mut i = 1;
    while i * i <= n { if n % i == 0 { a.push(i); if i * i != n { a.push(n/i); } } i+=1; }
    a.sort();
    a
}

// 素数列挙
pub fn primes(n: us) -> Vec<us> {
    if n <= 1 { return vec![]; }

    let mut sieve = vec![true; n+1];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2;
    while i * i <= n {
        if sieve[i] { for j in (i*i..=n).step_by(i) { sieve[j] = false; } }
        i += 1;
    }
    sieve.iter().enumerate().filter(|p|*p.1).map(|p|p.0).collect()
}

// 素因数分解する.
pub fn prime_fact(mut n: us) -> Vec<(us, us)> {
    let mut m = vec![];
    let mut d = 2;
    while d * d <= n {
        let mut e = 0;
        while n % d == 0 { e+=1; n /= d; }
        if e != 0 { m.push((d, e)); }
        d+=1;
    }

    if n != 1 { m.push((n, 1)); }
    m.sort();
    m
}

// ランレングス圧縮
pub fn runlength_encoding<T:PartialEq+Copy>(v: &[T]) -> Vec<(T, us)> {
    let mut a = Vec::new();
    for i in 0..v.len() {
        if i==0 || v[i-1]!=v[i] { a.push((v[i],0)) }
        a.last_mut().unwrap().1 += 1;
    }
    a
}

// xをn進数に変換する.
pub fn to_base_n(mut x: us, n: us) -> Vec<us> {
    let mut v = vec![];
    loop {
        v.push(x % n);
        x /= n;
        if x == 0 { break; }
    }
    v.reverse();
    v
}

// 拡張Euclidの互除法
// g=gcd(a,b)とし、ax + by = g となる(g, x, y)を返す
pub fn extgcd<N: ExPrimInt+::num::Zero+::num::One>(a: N, b: N) -> (N, N, N) {
    if b.is_zero() { (a, N::one(), N::zero()) }
    else {
        let (g, x, y) = extgcd(b, a % b);
        (g, y, x - a/b*y)
    }
}

// n^k mod m
pub fn powmod<N: ExPrimInt+::num::Zero+::num::One+ops::BitAnd<Output=N>+ops::Shr<Output=N>>(mut n: N, mut k: N, m: N) -> N {
    let one = N::one();
    let mut a = one;
    while k > N::zero() {
        if k & one == one { a *= n; a %= m; }
        n %= m; n *= n; n %= m;
        k = k >> one;
    }
    a
}

// Traveling Salesman Problem(巡回セールスマン問題)
// g[u][v] := u->vの距離 (gは隣接行列とする)
// st: 始点(複数指定可)
// return_to_home: 始点に戻るまで計算対象にする場合true
// 計算量: 頂点数をnとしてO(n^2*2^n)
// 全頂点を巡回してstに戻るときの最小コストは、return_to_home=trueで実行したときのdp[(1<<n)-1][st]で取得できる.
fn tsp(g: &Vec<Vec<i64>>, st: &[us], return_to_home: bool) -> Vec<Vec<i64>> {
    use itertools::iproduct;
    use crate:: chmin;

    let n = g.len();
    // dp[s][v] := 0から出発してsの頂点集合を巡回しvに到達する経路のうちの最短距離
    let mut dp = vec![vec![i64::INF; n]; 1<<n];
    let initial_bit = |st: us| { if return_to_home { 0 } else { 1<<st } };
    for &st in st { dp[initial_bit(st)][st] = 0; }
    for (s, u, v) in iproduct!(0..1<<n, 0..n, 0..n) {
        if s>>v&1==1 || dp[s][u]==i64::INF || g[u][v]==i64::INF { continue; }
        chmin!(dp[s|1<<v][v], dp[s][u]+g[u][v]);
    }
    dp
}

// パスカルの三角形によりnCkを計算. O(n^2)
pub struct Combination<N>(Vec<Vec<N>>);
impl<N: Copy+IntoT<us>+Zero+One+AddAssign> Combination<N> {
    pub fn new(n: us) -> Self {
        let mut m = vec![vec![N::zero(); n+1]; n+1];
        m[0][0] = N::one();
        for i in 0..n { for j in 0..n {
            add!(m[i+1][j],   m[i][j]);
            add!(m[i+1][j+1], m[i][j]);
        }}
        Self(m)
    }

    pub fn nk<T: IntoT<us>+Zero+Ord>(&self, n: T, k: T) -> N {
        if k < T::zero() || k > n { N::zero() } else { self.0[n.us()][k.us()] }
    }
}

// 2次元配列を右に90度回転させる
fn rot_right<T:Clone>(v: &[Vec<T>]) -> Vec<Vec<T>> {
    let (h, w) = (v.len(), v[0].len());
    let mut a = vec![vec![v[0][0].clone(); h]; w];
    for i in 0..h { for j in 0..w { a[j][h-1-i]=v[i][j].clone(); }}
    a
}

// 2次元配列を左に90度回転させる
fn rot_left<T:Clone>(v: &[Vec<T>]) -> Vec<Vec<T>> {
    let (h, w) = (v.len(), v[0].len());
    let mut a = vec![vec![v[0][0].clone(); h]; w];
    for i in 0..h { for j in 0..w { a[w-j-1][i]=v[i][j].clone(); }}
    a
}

pub fn on_thread<F: FnOnce()->()+Send+'static>(f: F) {
    // 再帰が深いなどスタックサイズが足りない場合はこのメソッドを利用する.
    std::thread::Builder::new()
        .stack_size(1024*1024*1024)
        .spawn(f)
        .unwrap()
        .join().unwrap();
}

pub fn gcd<N: ExPrimInt> (mut a: N, mut b: N) -> N {
    while b > N::zero() { let c = b; b = a % b; a = c; } a
}
pub fn lcm<N: ExPrimInt> (a: N, b: N) -> N {
    if a.is_zero() || b.is_zero() { N::zero() } else { a / gcd(a,b) * b }
}
pub fn floor_s<N: SimplePrimInt+Neg<Output=N>> (a: N, b: N) -> N {
    if a>=N::zero() { floor(a, b) } else { -ceil(-a, b) }
}
pub fn ceil_s<N: SimplePrimInt+Neg<Output=N>> (a: N, b: N) -> N {
    if a>=N::zero() { ceil(a, b) } else { -floor(-a, b) }
}
pub fn safe_mod<N: ExPrimInt> (n: N, m: N) -> N { (n % m + m) % m }
pub fn sumae<N: SimplePrimInt>(n: N, a: N, e: N) -> N { n * (a + e) / N::two() }
pub fn sumaed<N: SimplePrimInt>(a: N, e: N, d: N) -> N { ((e - a) / d) * (a + e) / N::two() }
pub fn sumad<N: SimplePrimInt>(n: N, a: N, d: N) -> N { n * (N::two() * a + (n - N::one()) * d) / N::two() }
pub fn ndigits<N:IntoT<us>>(n: N) -> us {
    use superslice::*;
    const POW10: [us; 20] = {
        let mut a = [1; 20];
        let mut i = 0;
        while i < 19 { a[i+1] = a[i] * 10; i += 1; }
        a
    };
    POW10.upper_bound(&n.into_t())
}

pub fn factorial<N: Copy+Eq+Mul<N>+Sub<Output=N>+One>(n: N) -> N {
    if n.is_one() { N::one() } else { n * factorial(n - N::one()) }
}

pub fn bin_search_f64(mut ok: f64, mut ng: f64, f: impl Fn(f64)->bool, mut iter: us) -> f64 {
    while iter > 0 {
        iter -= 1;
        let m = (ok + ng) / 2.;
        if f(m) { ok = m; } else { ng = m; }
    }
    ok
}

// 三分探索
fn ternary_search<T: PartialOrd>(mut l: i64, mut r: i64, f: impl Fn(i64)->T) -> (i64, i64) {
    while r - l > 2 {
        let m1 = (l * 2 + r) / 3;
        let m2 = (l + r * 2) / 3;
        if f(m1) > f(m2) { l = m1; } else { r = m2; }
    }
    (l, r)
}

// io
// インタラクティブ問題ではこれをinputに渡す
// let mut src = from_stdin();
// input! {from &mut src, n: usize}
pub fn from_stdin() -> impl proconio::source::Source<io::BufReader<io::Stdin>> {
    proconio::source::line::LineSource::new(io::BufReader::new(io::stdin()))
}

// CAP(IGNORE_BELOW)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primes() {
        assert_eq!(primes(0),  vec![]);
        assert_eq!(primes(1),  vec![]);
        assert_eq!(primes(2),  vec![2]);
        assert_eq!(primes(50), vec![2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]);
        assert_eq!(primes(53), vec![2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53]);
    }

    #[test]
    fn test_runlength_encoding() {
        assert_eq!(runlength_encoding(&vec!['a','a','b','a','a','c']), vec![('a',2),('b',1),('a',2),('c',1)]);
    }


}