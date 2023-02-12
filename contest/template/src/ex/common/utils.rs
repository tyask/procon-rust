#![allow(dead_code)]
use crate::common::*;

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
pub fn prime_fact(mut n: us) -> bmap<us, us> {
    let mut m = bmap::new();
    let mut d = 2;
    while d * d <= n {
        let mut e = 0;
        while n % d == 0 { e+=1; n /= d; }
        if e != 0 { m.insert(d, e); }
        d+=1;
    }

    if n != 1 { m.insert(n, 1); }
    m
}

// 座標圧縮
pub fn compress<T:Clone+PartialEq+Ord>(v: &Vec<T>) -> Vec<us> {
    use superslice::Ext;
    use itertools::Itertools;
    let t = v.iter().cloned().sorted().dedup().collect_vec();
    v.iter().map(|x|t.lower_bound(x)).collect_vec()
}

// ランレングス圧縮
pub fn runlength_encoding<T:PartialEq+Copy>(v: &Vec<T>) -> Vec<(T, us)> {
    let mut a = Vec::new();
    for i in 0..v.len() {
        if i==0 || v[i-1]!=v[i] { a.push((v[i],0)) }
        let n = a.len(); a[n-1].1 += 1;
    }
    a
}

// 2次元配列を右に90度回転させる
pub fn rot<T:Copy>(g: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let (h, w) = (g.len(), g[0].len());
    let mut a = vec![vec![g[0][0]; h]; w];
    for i in 0..h { for j in 0..w { a[j][h-1-i]=g[i][j]; }}
    a
}

pub fn to_base_n(mut x: us, n: us) {
    let mut v = vec![];
    while x > 0 { v.push(x%n); x/=n; }
    v.reverse()
}

// 2部グラフの色を0,1で塗り分ける. 2部グラフでない場合Err.
// g: グラフを表す隣接リスト
fn colorize_bipartite(g: &Vec<Vec<us>>) -> Result<Vec<is>,()> {
    let mut col = vec![-1; g.len()];
    fn dfs(g: &Vec<Vec<us>>, col: &mut Vec<is>, v: us, c: is) -> bool {
        if col[v] >= 0 { return true; }
        col[v] = c;
        for &n in &g[v] {
            if col[n] == c || col[n] < 0 && !dfs(g, col, n, 1-c) { return false; }
        }
        true
    }

    for v in 0..g.len() { if !dfs(&g, &mut col, v, 0) { return Err(()); } }
    Ok(col)
}

// パスカルの三角形によりnCkを計算. O(n^2)
pub struct Combination { m: Vec<Vec<us>>, }
impl Combination {
    pub fn new(n: impl IntoT<us>) -> Self {
        let n = n.into_t();
        let mut m = vec![vec![0; n+1]; n+1];
        m[0][0] = 1;
        for i in 0..n { for j in 0..n {
            m[i+1][j]   += m[i][j];
            m[i+1][j+1] += m[i][j];
        }}
        Self { m }
    }

    pub fn nk<N: SimplePrimInt + IntoT<us> + FromT<us>>(&self, n: N, k: N) -> us {
        if k < N::from_t(0) || k > n { 0 } else { self.m[n.into_t()][k.into_t()] }
    }
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
    fn test_compress() {
        assert_eq!(compress(&vec![1,3,8,4]), vec![0,1,3,2]);
    }

    #[test]
    fn test_runlength_encoding() {
        assert_eq!(runlength_encoding(&vec!['a','a','b','a','a','c']), vec![('a',2),('b',1),('a',2),('c',1)]);
    }


}