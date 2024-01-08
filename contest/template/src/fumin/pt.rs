#![allow(dead_code)]
use std::{*, ops::*, iter::Sum};
use num_traits::Signed;

use crate::common::*;

// Pt
#[derive(Debug,Copy,Clone,PartialEq,Eq,Hash,Default)]
pub struct Pt<N> { pub x: N, pub y: N }

impl<N> Pt<N> {
    pub fn new(x: impl IntoT<N>, y: impl IntoT<N>) -> Pt<N> { Pt{x:x.into_t(), y:y.into_t()} }
    pub fn of(x: N, y: N) -> Pt<N> { Pt{x:x, y:y} }
    pub fn tuple(self) -> (N, N) { (self.x, self.y) }
}
impl<N: SimplePrimInt> Pt<N> {
    pub fn norm2(self) -> N   { self.x * self.x + self.y * self.y }
    pub fn on(self, h: Range<N>, w: Range<N>) -> bool { h.contains(&self.x) && w.contains(&self.y) }
    pub fn manhattan_distance(self, p: Pt<N>) -> N { abs_diff(self.x, p.x) + abs_diff(self.y, p.y) }
}
impl<N: SimplePrimInt+FromT<i64>+ToF64> Pt<N> {
    pub fn norm(self) -> f64 { self.norm2().f64().sqrt() }
}
impl<N: Wrapping> Wrapping for Pt<N> {
    fn wrapping_add(self, a: Self) -> Self { Self::of(self.x.wrapping_add(a.x), self.y.wrapping_add(a.y)) }
}
impl Pt<us> {
    pub const C4: [char; 4] = ['R', 'L', 'D', 'U'];
    pub const D4: [Self; 4] = [
        Self{x:0,y:1},Self{x:0,y:!0},Self{x:1,y:0},Self{x:!0,y:0}
        ];
    pub const D8: [Self; 8] = [
        Self{x:0,y:1},Self{x:0,y:!0},Self{x:1,y:0},Self{x:!0,y:0},
        Self{x:1,y:1},Self{x:1,y:!0},Self{x:!0,y:1},Self{x:!0,y:!0},
        ];
    pub fn wrapping_sub(self, a: Self) -> Self { Self::of(self.x.wrapping_sub(a.x), self.y.wrapping_sub(a.y)) }
    pub fn di(self, b: Pt<us>) -> us { Self::D8.pos(&b.wrapping_sub(self)).unwrap() }
    pub fn dic(self, b: Pt<us>) -> char { Self::C4[Self::di(self,b)] }
}


impl<N: Copy + Signed> Pt<N> {
    // 外積 (ベクトルself, vからなる平行四辺形の符号付面積)
    pub fn cross(&self, v: Pt<N>) -> N {
        self.x * v.y - self.y * v.x
    }

    // couter cross wise (ベクトルself, vが反時計回りかどうか)
    pub fn ccw(&self, v: Pt<N>) -> i32 {
        let a = self.cross(v);
        if a.is_positive() { 1 } // ccw
        else if a.is_negative() { -1 } // cw
        else { 0 } // colinear
    }
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
impl<T> From<(T, T)> for Pt<T> {
    fn from(t: (T, T)) -> Self { Self::of(t.0, t.1) }
}

// CAP(IGNORE_BELOW)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_us() {
        type P = Pt<us>;

        assert_eq!(P::di(P::new(2,3), P::new(2,4)), 0);
        assert_eq!(P::di(P::new(2,3), P::new(2,2)), 1);
        assert_eq!(P::di(P::new(2,3), P::new(3,3)), 2);
        assert_eq!(P::di(P::new(2,3), P::new(1,3)), 3);

        assert_eq!(P::dic(P::new(2,3), P::new(2,4)), 'R');
        assert_eq!(P::dic(P::new(2,3), P::new(2,2)), 'L');
        assert_eq!(P::dic(P::new(2,3), P::new(3,3)), 'D');
        assert_eq!(P::dic(P::new(2,3), P::new(1,3)), 'U');
    }
}