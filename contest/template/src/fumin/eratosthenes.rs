#![allow(dead_code)]
use std::ops::{AddAssign, SubAssign};
use crate::{common::*, add, sub};

// エラトステネスの篩
pub struct Eratosthenes {
    pub is_prime: Vec<bool>, // 素数の場合true
    pub primes: Vec<us>,
    pub min_factor: Vec<us>,   // i:=iを割り切る最小の素数

    // メビウス関数
    // i=1 のとき1
    // iがある素数pで2回以上割り切れるとき0 (12=2^2*3など)
    // iに含まれる素因数の指数の最大値が1のとき、素因数の個数をkとして、(-1)^k
    pub mobius: Vec<i64>,
}

impl Eratosthenes {
    pub fn new(n: us) -> Self {
        let mut is_prime = vec![true; n+1];
        let mut min_factor = vec![us::INF; n+1];
        let mut mobius = vec![1; n+1];

        is_prime[0] = false;
        is_prime[1] = false;
        min_factor[1] = 1;

        for p in 2..=n {
            if !is_prime[p] { continue; }
            min_factor[p] = p;
            mobius[p] = -1; // 素数は-1

            for q in (p*2..=n).step_by(p) {
                is_prime[q] = false;
                if min_factor[q] == us::INF { min_factor[q] = p; }
                mobius[q] = if (q/p)%p==0 { 0 } else { -mobius[q] };
            }
        }

        let primes = (2..=n).into_iter().filter(|&i|is_prime[i]).cv();

        Self { is_prime, primes, min_factor, mobius }
    }

    pub fn iter_primes(&self, n: us) -> impl Iterator<Item=us> + '_ {
        self.primes.iter().cloned().take_while(move|&i|i<=n)
    }

    // 高速素因数分解
    // Vec(素因子,指数)を返す.
    // 計算量はO(logN)
    pub fn factorize(&self, n: us) -> Vec<(us, us)> {
        let mut res = vec![];
        let mut n = n;
        while n > 1 {
            let p = self.min_factor[n];
            let mut e = 0;
            while self.min_factor[n] == p { n /= p; e += 1; }
            res.push((p, e));
        }
        res
    }

    // 高速約数列挙
    // Nの約数をの個数をMとして計算量はO(M)
    pub fn divisors(&self, n: us) -> Vec<us> {
        let mut res = vec![1];
        for (p, e) in self.factorize(n) {
            let mut v = 1;
            for x in res.clone() {
                for _ in 0..e { v *= p; res.push(x * v); }
            }
        }
        res
    }

    // (約数系の)高速ゼータ変換
    // nの関数f(n)について、f(1),f(2),..f(N)が与えられているとき、F(n)を以下のように定義する.
    // F(n) = N以下のnの倍数をiとしたときのf(i)の合計
    // 1~NについてのF(n)を計算する。 計算量はO(NloglogN)
    pub fn fast_zeta<N: Clone + AddAssign>(&self, f: &Vec<N>) -> Vec<N> {
        let n = f.len() - 1;
        let mut res = f.clone();
        for p in self.iter_primes(n) {
            for k in (1..=n/p).rev() { add!(res[k], res[k * p].clone()); }
        }
        res
    }

    // (約数系の)高速メビウス変換
    // 高速ゼータ変換の逆変換を行う.
    // 計算量はO(NloglogN)
    pub fn fast_mobius<N: Clone + SubAssign>(&self, f: &Vec<N>) -> Vec<N> {
        let n = f.len() - 1;
        let mut res = f.clone();
        for p in self.iter_primes(n) {
            let mut k = 1;
            while k * p <= n { sub!(res[k], res[k*p].clone()); k += 1; }
        }
        res
    }

    pub fn eulers_phi(&self, n: us) -> i64 {
        let s = self.iter_primes(n)
            .map(|p|self.mobius[p]*(n/p).i64())
            .sum::<i64>();
         n.i64() - s
    }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use super::*;

    #[test]
    fn test_zeta_mobius() {
        let e = Eratosthenes::new(12);
        let f = Vec::from_iter(0..=12);
        let f2 = e.fast_mobius(&e.fast_zeta(&f));

        assert_eq!(f, f2);
    }

    #[test]
    fn test_eulers_phi() {
        let e = Eratosthenes::new(12);
        let a = (1..=12).into_iter().map(|i|e.eulers_phi(i)).cv();
        eprintln!("a = {:?}", a);
    }

}