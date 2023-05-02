#![allow(dead_code)]
use std::{*, ops::*, iter::Sum};
use crate::common::*;

// Pt
#[derive(Debug,Copy,Clone,PartialEq,Eq,Hash,PartialOrd,Ord,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N> Pt<N> {
    pub fn new(x: impl IntoT<N>, y: impl IntoT<N>) -> Pt<N> { Pt{x:x.into_t(), y:y.into_t()} }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
}
impl<N: SimplePrimInt> Pt<N> {
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn on(self, h: Range<N>, w: Range<N>) -> bool { h.contains(&self.x) && w.contains(&self.y) }
}
impl<N: SimplePrimInt+FromT<is>> Pt<N> {
    pub fn dir4() -> Vec<Pt<N>> {
        vec![Pt::is(0,1), Pt::is(0,!0), Pt::is(1,0), Pt::is(!0,0)]
    }
    pub fn dir8() -> Vec<Pt<N>> {
        vec![
            Pt::is(0,1), Pt::is(0,!0), Pt::is(1, 0), Pt::is(!0, 0),
            Pt::is(1,1), Pt::is(1,!0), Pt::is(!0,1), Pt::is(!0,!0)
            ]
    }
    fn is(x: is, y: is) -> Pt<N> { Self::of(N::from_t(x), N::from_t(y)) }
}
impl<N: SimplePrimInt+FromT<is>+ToF64> Pt<N> {
    pub fn norm(self)  -> f64 { self.norm2().f64().sqrt() }
}
impl<N: Wrapping> Wrapping for Pt<N> {
    fn wraping_add(self, a: Self) -> Self { Self::of(self.x.wraping_add(a.x), self.y.wraping_add(a.y)) }
}

pub type Radian = f64;
impl Pt<f64> {
    pub fn rot(self, r: Radian) -> Pt<f64> {
        let (x, y) = (self.x, self.y);
        Self::new(r.cos()*x-r.sin()*y, r.sin()*x+r.cos()*y) // 反時計回りにr度回転(rはradian)
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
impl<N: SimplePrimInt+FromT<is>> Neg    for Pt<N> { type Output = Pt<N>; fn neg(mut self) -> Self::Output { self *= N::from_t(-1); self } }
impl<N: SimplePrimInt+Default> Sum      for Pt<N> { fn sum<I: Iterator<Item=Self>>(iter: I) -> Self { iter.fold(Self::default(), |a, b| a + b) } }

impl<N: SimplePrimInt+FromT<is>+proconio::source::Readable<Output=N>+IntoT<N>> proconio::source::Readable for Pt<N> {
    type Output = Pt<N>;
    fn read<R: io::BufRead, S: proconio::source::Source<R>>(source: &mut S) -> Self::Output {
        Pt::new(N::read(source), N::read(source))
    }
}
