
#[allow(dead_code)]
pub mod modint {
use std::{ops::*, marker::PhantomData, io::BufRead, fmt::{Display, Formatter, Debug}, iter::Sum};
use proconio::source::{Readable, Source};

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
    phantom: PhantomData<M>,
}

impl<M: Modules> Modint<M> {
    const MOD: usize = M::MOD;
    pub fn new(val: usize) -> Modint<M> { let m = Self::MOD; Modint{val: (val%m+m)%m, phantom: PhantomData} }
    pub fn zero() -> Modint<M> { Self::new(0) }
    pub fn one()  -> Modint<M> { Self::new(1) }
    pub fn pow(self, t: usize) -> Modint<M> {
        if t == 0 { return Self::new(1); }
        let mut a = self.pow(t>>1);
        a *= a;
        if t & 1 == 1 { a *= self; }
        a
    }
    pub fn inv(self) -> Modint<M> { self.pow(Self::MOD-2) }
    fn normalize(&mut self) { if self.val >= Self::MOD { self.val -= Self::MOD; } }
}

impl<M: Modules> Default  for Modint<M> { fn default() -> Self { Self::new(0) } }
impl<M: Modules> Display  for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.val) } }
impl<M: Modules> Debug    for Modint<M> { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { Display::fmt(&self, f) } }
impl<M: Modules> Readable for Modint<M> { type Output = Modint<M>; fn read<R: BufRead, S: Source<R>>(source: &mut S) -> Self::Output { Modint::new(usize::read(source)) } }
impl<M: Modules> Sum      for Modint<M> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::zero(), |a, b| a + b) } }

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

}


#[allow(dead_code)]
pub mod combination {
use super::modint::{Modint, Modules};

// combination
struct Combination<M> { facts: Vec<Modint<M>>, ifacts: Vec<Modint<M>> }
impl<M: Modules> Combination<M> {
    pub fn new(n: usize) -> Combination<M> {
        assert!(n < M::MOD);
        let mut facts  = vec![Modint::<M>::zero(); n+1];
        let mut ifacts = vec![Modint::<M>::zero(); n+1];
        facts[0] = Modint::<M>::one();
        for i in 1..n+1 { facts[i] = facts[i-1]*i; }
        ifacts[n] = facts[n].inv();
        for i in (1..n+1).rev() { ifacts[i-1] = ifacts[i]*i; }
        Combination{facts: facts, ifacts: ifacts}
    }

    // nCk
    pub fn comb(&self, n: usize, k: usize) -> Modint<M> {
        assert!(n < M::MOD);
        if n < k { return Modint::<M>::zero(); }
        self.fact(n) * self.ifact(k) * self.ifact(n-k)
    }
    // k!
    pub fn fact(&self, k: usize)  -> Modint<M> { self.facts[k] }
    pub fn ifact(&self, k: usize) -> Modint<M> { self.ifacts[k] }
}

}

#[cfg(test)]
mod tests {
    use crate::ex::modint::modint::*;
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