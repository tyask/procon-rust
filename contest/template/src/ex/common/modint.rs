#![allow(dead_code)]
use std::{ops::*, marker::PhantomData, io::BufRead, fmt::{Display, Formatter, Debug}, iter::Sum};
use proconio::source::{Readable, Source};
use crate::common::*;

pub type Modint1000000007 = Modint<Modules1000000007>;
pub type Modint998244353  = Modint<Modules998244353>;

pub trait Modules: Clone + Copy { const MOD: usize; }
#[derive(Clone, Copy, PartialEq)] pub enum Modules1000000007 { }
#[derive(Clone, Copy, PartialEq)] pub enum Modules998244353 { }
impl Modules for Modules1000000007 { const MOD: usize = 1_000_000_007; }
impl Modules for Modules998244353  { const MOD: usize =   998_244_353; }

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Modint<M> {
    val: usize,
    phantom: PhantomData<dyn Fn() -> M>,
}

impl<M: Modules> Modint<M> {
    pub const MOD: usize = M::MOD;
    pub fn new(val: usize) -> Modint<M> { let m = Self::MOD; Modint{val: (val%m+m)%m, phantom: PhantomData} }
    pub fn pow(self, t: usize) -> Modint<M> {
        if t == 0 { return Self::new(1); }
        let mut a = self.pow(t>>1);
        a *= a;
        if t & 1 == 1 { a *= self; }
        a
    }
    pub fn inv(self) -> Modint<M> { self.pow((Self::MOD-2) as usize) }
    fn normalize(&mut self) { if self.val >= Self::MOD { self.val -= Self::MOD; } }
}

impl<M: Modules> Default  for Modint<M> { fn default() -> Self { Self::from_t(0) } }
impl<M: Modules> Display  for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.val) } }
impl<M: Modules> Debug    for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { Display::fmt(&self, f) } }
impl<M: Modules> Readable for Modint<M> { type Output = Modint<M>; fn read<R: BufRead, S: Source<R>>(source: &mut S) -> Self::Output { Modint::new(usize::read(source)) } }
impl<M: Modules> Sum      for Modint<M> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::from_t(0), |a, b| a + b) } }

impl<M: Modules, N: IntoT<usize>> FromT<N> for Modint<M> { fn from_t(n: N) -> Self { Self::new(n.into_t()) } }
impl<M: Modules, N: IntoT<usize>> From<N>  for Modint<M> { fn from(n: N)   -> Self { Self::new(n.into_t()) } }

impl<M: Modules> AddAssign for Modint<M> { fn add_assign(&mut self, rhs: Self) { self.val += rhs.val; self.normalize(); } }
impl<M: Modules> SubAssign for Modint<M> { fn sub_assign(&mut self, rhs: Self) { self.val += Self::MOD-rhs.val; self.normalize(); } }
impl<M: Modules> MulAssign for Modint<M> { fn mul_assign(&mut self, rhs: Self) { self.val *= rhs.val; self.val %= Self::MOD; } }
impl<M: Modules> DivAssign for Modint<M> { fn div_assign(&mut self, rhs: Self) { *self *= rhs.inv(); } }
impl<M: Modules> AddAssign<usize> for Modint<M> { fn add_assign(&mut self, rhs: usize) { *self += Self::new(rhs); } }
impl<M: Modules> SubAssign<usize> for Modint<M> { fn sub_assign(&mut self, rhs: usize) { *self -= Self::new(rhs); } }
impl<M: Modules> MulAssign<usize> for Modint<M> { fn mul_assign(&mut self, rhs: usize) { *self *= Self::new(rhs); } }
impl<M: Modules> DivAssign<usize> for Modint<M> { fn div_assign(&mut self, rhs: usize) { *self /= Self::new(rhs); } }

#[macro_export] macro_rules! impl_op {
    ($($t:ty),*) => {$(
        impl<M: Modules> Add<$t> for Modint<M> { type Output = Modint<M>; fn add(mut self, rhs: $t) -> Self::Output { self += rhs; self } }
        impl<M: Modules> Sub<$t> for Modint<M> { type Output = Modint<M>; fn sub(mut self, rhs: $t) -> Self::Output { self -= rhs; self } }
        impl<M: Modules> Mul<$t> for Modint<M> { type Output = Modint<M>; fn mul(mut self, rhs: $t) -> Self::Output { self *= rhs; self } }
        impl<M: Modules> Div<$t> for Modint<M> { type Output = Modint<M>; fn div(mut self, rhs: $t) -> Self::Output { self /= rhs; self } }
    )*}
}
impl_op!(Modint<M>, usize);

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