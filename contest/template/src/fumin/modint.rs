#![allow(dead_code)]
use std::{ops::*, io::BufRead, fmt::{Display, Formatter, Debug}, iter::Sum};
use num::{Zero, One};
use proconio::source::{Readable, Source};
use crate::common::*;

pub type Modint1000000007 = Modint<1000000007>;
pub type Modint998244353  = Modint<998244353>;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Modint<const M: i64> {
    val: i64,
}

impl<const M: i64> Modint<M> {
    pub fn new(v: impl IntoT<i64>) -> Modint<M> {
        let (m, v) = (M, v.into_t());
        Modint{val: (v%m+m)%m}
    }
    pub fn pow(self, t: impl IntoT<i64>) -> Modint<M> {
        let t = t.into_t();
        if t == 0 { return Self::new(1); }
        let mut a = self.pow(t>>1);
        a *= a;
        if t & 1 == 1 { a *= self; }
        a
    }
    pub fn inv(self) -> Modint<M> { self.pow((M-2) as i64) }
    fn normalize(&mut self) { if self.val >= M { self.val -= M; } }
}

impl<const M: i64> Default  for Modint<M> { fn default() -> Self { Self::from_t(0) } }
impl<const M: i64> Display  for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.val) } }
impl<const M: i64> Debug    for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { Display::fmt(&self, f) } }
impl<const M: i64> Readable for Modint<M> { type Output = Modint<M>; fn read<R: BufRead, S: Source<R>>(source: &mut S) -> Self::Output { Modint::new(i64::read(source)) } }
impl<const M: i64> Sum      for Modint<M> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::from_t(0), |a, b| a + b) } }

impl<const M: i64, N: IntoT<i64>> FromT<N> for Modint<M> { fn from_t(n: N) -> Self { Self::new(n.into_t()) } }
impl<const M: i64, N: IntoT<i64>> From<N>  for Modint<M> { fn from(n: N)   -> Self { Self::new(n.into_t()) } }

impl<const M: i64> AddAssign for Modint<M> { fn add_assign(&mut self, rhs: Self) { self.val += rhs.val; self.normalize(); } }
impl<const M: i64> SubAssign for Modint<M> { fn sub_assign(&mut self, rhs: Self) { self.val += M-rhs.val; self.normalize(); } }
impl<const M: i64> MulAssign for Modint<M> { fn mul_assign(&mut self, rhs: Self) { self.val *= rhs.val; self.val %= M; } }
impl<const M: i64> DivAssign for Modint<M> { fn div_assign(&mut self, rhs: Self) { *self *= rhs.inv(); } }
impl<const M: i64> Add<Self> for Modint<M> { type Output = Self; fn add(mut self, rhs: Self) -> Self::Output { self += rhs; self } }
impl<const M: i64> Sub<Self> for Modint<M> { type Output = Self; fn sub(mut self, rhs: Self) -> Self::Output { self -= rhs; self } }
impl<const M: i64> Mul<Self> for Modint<M> { type Output = Self; fn mul(mut self, rhs: Self) -> Self::Output { self *= rhs; self } }
impl<const M: i64> Div<Self> for Modint<M> { type Output = Self; fn div(mut self, rhs: Self) -> Self::Output { self /= rhs; self } }

impl<const M: i64, N: IntoT<i64>> AddAssign<N> for Modint<M> { fn add_assign(&mut self, rhs: N) { *self += Self::new(rhs); } }
impl<const M: i64, N: IntoT<i64>> SubAssign<N> for Modint<M> { fn sub_assign(&mut self, rhs: N) { *self -= Self::new(rhs); } }
impl<const M: i64, N: IntoT<i64>> MulAssign<N> for Modint<M> { fn mul_assign(&mut self, rhs: N) { *self *= Self::new(rhs); } }
impl<const M: i64, N: IntoT<i64>> DivAssign<N> for Modint<M> { fn div_assign(&mut self, rhs: N) { *self /= Self::new(rhs); } }
impl<const M: i64, N: IntoT<i64>> Add<N> for Modint<M> { type Output = Self; fn add(mut self, rhs: N) -> Self::Output { self += rhs; self } }
impl<const M: i64, N: IntoT<i64>> Sub<N> for Modint<M> { type Output = Self; fn sub(mut self, rhs: N) -> Self::Output { self -= rhs; self } }
impl<const M: i64, N: IntoT<i64>> Mul<N> for Modint<M> { type Output = Self; fn mul(mut self, rhs: N) -> Self::Output { self *= rhs; self } }
impl<const M: i64, N: IntoT<i64>> Div<N> for Modint<M> { type Output = Self; fn div(mut self, rhs: N) -> Self::Output { self /= rhs; self } }

impl<const M: i64> Zero for Modint<M> {
    fn zero() -> Self { Self::new(0) }
    fn is_zero(&self) -> bool { *self == Self::zero() }
}
impl<const M: i64> One for Modint<M> {
    fn one()  -> Self { Self::new(1) }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;
    type Mint = Modint1000000007;

    #[test]
    fn test_modint() {
        let a = Mint::new(1);
        let b = Mint::new(1000000008);
        let e = Mint::new(1000000007);

        assert_eq!(a,     b);
        assert_eq!(a+e,   b);
        assert_eq!(a + 1, Mint::new(2));
    }
}