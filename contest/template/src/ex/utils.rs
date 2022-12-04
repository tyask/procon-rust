#[allow(dead_code)]
pub mod utils {

use std::collections::BTreeMap;

// 約数を列挙する
pub fn divisors(n: usize) -> Vec<usize> {
    let mut a = Vec::new();
    let mut i = 1;
    while i * i <= n { if n % i == 0 { a.push(i); if i * i != n { a.push(n/i); } } i+=1; }
    a.sort();
    a
}

// 素因数分解する.
pub fn prime_fact(mut n: usize) -> BTreeMap<usize, usize> {
    let mut m = BTreeMap::new();
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
pub fn compress<T:Clone+PartialEq+Ord>(v: &Vec<T>) -> Vec<usize> {
    use superslice::Ext;
    let mut t = v.to_vec(); t.sort(); t.dedup();
    v.iter().map(|x| t.lower_bound(x)).collect::<Vec<_>>()
}

// ランレングス圧縮
pub fn runlength_encoding<T:PartialEq+Copy>(v: &Vec<T>) -> Vec<(T, usize)> {
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
}

#[cfg(test)]
mod tests {
    use crate::ex::utils::utils::*;

    #[test]
    fn test_compress() {
        assert_eq!(compress(&vec![1,3,8,4]), vec![0,1,3,2]);
    }

    #[test]
    fn test_runlength_encoding() {
        assert_eq!(runlength_encoding(&vec!['a','a','b','a','a','c']), vec![('a',2),('b',1),('a',2),('c',1)]);
    }


}