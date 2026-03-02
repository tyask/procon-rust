#![allow(dead_code)]
use crate::common::*;

pub(crate) trait BitWord:
    Clone
    + Copy
    + std::fmt::Binary
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + std::hash::Hash
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Shl<us, Output = Self>
    + std::ops::BitAndAssign
    + std::ops::BitOrAssign
{
    const BITS: us;
    fn zero() -> Self;
    fn one() -> Self;
    fn max_value() -> Self;
    fn count_ones(self) -> u32;
    fn trailing_zeros(self) -> u32;
}

impl BitWord for u32 {
    const BITS: us = std::mem::size_of::<u32>() * 8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn max_value() -> Self { u32::MAX }
    fn count_ones(self) -> u32 { u32::count_ones(self) }
    fn trailing_zeros(self) -> u32 { u32::trailing_zeros(self) }
}

impl BitWord for u8 {
    const BITS: us = std::mem::size_of::<u8>() * 8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn max_value() -> Self { u8::MAX }
    fn count_ones(self) -> u32 { u8::count_ones(self) }
    fn trailing_zeros(self) -> u32 { u8::trailing_zeros(self) }
}

impl BitWord for u16 {
    const BITS: us = std::mem::size_of::<u16>() * 8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn max_value() -> Self { u16::MAX }
    fn count_ones(self) -> u32 { u16::count_ones(self) }
    fn trailing_zeros(self) -> u32 { u16::trailing_zeros(self) }
}

impl BitWord for u64 {
    const BITS: us = std::mem::size_of::<u64>() * 8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn max_value() -> Self { u64::MAX }
    fn count_ones(self) -> u32 { u64::count_ones(self) }
    fn trailing_zeros(self) -> u32 { u64::trailing_zeros(self) }
}

impl BitWord for u128 {
    const BITS: us = std::mem::size_of::<u128>() * 8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    fn max_value() -> Self { u128::MAX }
    fn count_ones(self) -> u32 { u128::count_ones(self) }
    fn trailing_zeros(self) -> u32 { u128::trailing_zeros(self) }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash)]
pub struct BitSetWord<W> {
    b: W,
    mask: W,
}

pub type BitSet8 = BitSetWord<u8>;
pub type BitSet16 = BitSetWord<u16>;
pub type BitSet32 = BitSetWord<u32>;
pub type BitSet64 = BitSetWord<u64>;
pub type BitSet128 = BitSetWord<u128>;

impl<W: BitWord> BitSetWord<W> {
    pub fn new(n: us) -> Self { Self::with_capacity(n) }
    pub fn with_capacity(n: us) -> Self {
        assert!(n <= W::BITS);
        let mask = if n == W::BITS {
            W::max_value()
        } else if n == 0 {
            W::zero()
        } else {
            (W::one() << n) - W::one()
        };
        Self { b: W::zero(), mask }
    }

    pub fn count_ones(&self) -> us { self.b.count_ones() as us }
    pub fn iter_ones(&self) -> impl Iterator<Item = us> {
        let mut b = self.b;
        std::iter::from_fn(move || {
            if b == W::zero() {
                None
            } else {
                let i = b.trailing_zeros() as us;
                b &= b - W::one();
                Some(i)
            }
        })
    }

    pub fn set(&mut self, i: us, f: bool) {
        assert!(i < W::BITS);
        if f { self.b |= W::one() << i; } else { self.b &= !(W::one() << i) }
        self.b &= self.mask;
    }
}

impl<W: BitWord> std::fmt::Debug for BitSetWord<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:b}", self.b)
    }
}

impl<W: BitWord> std::ops::BitAnd for BitSetWord<W> {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self { b: self.b & rhs.b, mask: self.mask }
    }
}
impl<W: BitWord> std::ops::BitOr for BitSetWord<W> {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self { b: (self.b | rhs.b) & self.mask, mask: self.mask }
    }
}
impl<W: BitWord> std::ops::Not for BitSetWord<W> {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self { b: (!self.b) & self.mask, mask: self.mask }
    }
}
impl<W: BitWord> std::ops::BitAndAssign for BitSetWord<W> {
    fn bitand_assign(&mut self, rhs: Self) {
        self.b &= rhs.b;
    }
}
impl<W: BitWord> std::ops::BitOrAssign for BitSetWord<W> {
    fn bitor_assign(&mut self, rhs: Self) {
        self.b = (self.b | rhs.b) & self.mask;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, std::hash::Hash)]
pub struct BitSetVec<const BLOCK: us> {
    b: [BitSet128; BLOCK],
    block: us,
}

impl<const BLOCK: us> BitSetVec<BLOCK> {
    const M: us = std::mem::size_of::<u128>() * 8;

    pub fn new(n: us) -> Self { Self::with_capacity(n) }
    pub fn with_capacity(n: us) -> Self {
        assert!(n <= BLOCK * Self::M);

        let mut b = [BitSet128::with_capacity(Self::M); BLOCK];
        let block = if n == 0 { 0 } else { ((n - 1) >> 7) + 1 };
        if block == 0 {
            b.fill(BitSet128::with_capacity(0));
            return Self { b, block };
        }

        let last_bits = n - (block - 1) * Self::M;
        b[block - 1] = BitSet128::with_capacity(last_bits);
        b[block..].fill(BitSet128::with_capacity(0));
        Self { b, block }
    }

    pub fn bitand(&self, other: &Self) -> Self {
        let mut t = *self;
        t.bitand_assign(other);
        t
    }
    pub fn bitor(&self, other: &Self) -> Self {
        let mut t = *self;
        t.bitor_assign(other);
        t
    }
    pub fn not(&self) -> Self {
        let mut t = *self;
        for i in 0..self.block { t.b[i] = !self.b[i]; }
        t
    }
    pub fn count_ones(&self) -> us {
        self.b[..self.block].iter().map(|b| b.count_ones()).sum::<us>()
    }
    pub fn iter_ones(&self) -> impl Iterator<Item = us> + '_ {
        (0..self.block).flat_map(move |i| self.b[i].iter_ones().map(move |bi| i * Self::M + bi))
    }
    pub fn bitand_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i] &= other.b[i]; }
    }
    pub fn bitor_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i] |= other.b[i]; }
    }
    pub fn set(&mut self, i: us, f: bool) {
        assert!(i < self.block * Self::M);
        self.b[i >> 7].set(i & (Self::M - 1), f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset32_ops() {
        let mut a = BitSet32::new(8);
        a.set(1, true);
        a.set(3, true);
        let mut b = BitSet32::new(8);
        b.set(1, true);
        b.set(2, true);

        assert_eq!((a & b).iter_ones().collect::<Vec<_>>(), vec![1]);
        assert_eq!((a | b).iter_ones().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!((!a).iter_ones().collect::<Vec<_>>(), vec![0, 2, 4, 5, 6, 7]);

        let mut c = a;
        c &= b;
        assert_eq!(c.iter_ones().collect::<Vec<_>>(), vec![1]);
        let mut d = a;
        d |= b;
        assert_eq!(d.iter_ones().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn bitset64_and_128_set_ones() {
        let mut s64 = BitSet64::new(64);
        s64.set(0, true);
        s64.set(63, true);
        assert_eq!(s64.count_ones(), 2);
        assert_eq!(s64.iter_ones().collect::<Vec<_>>(), vec![0, 63]);

        let mut s128 = BitSet128::new(128);
        s128.set(5, true);
        s128.set(127, true);
        assert_eq!(s128.count_ones(), 2);
        assert_eq!(s128.iter_ones().collect::<Vec<_>>(), vec![5, 127]);
    }

    #[test]
    fn bitset_vec_crosses_128_boundary() {
        let mut a = BitSetVec::<2>::new(256);
        a.set(127, true);
        a.set(128, true);
        a.set(200, true);

        let mut b = BitSetVec::<2>::new(256);
        b.set(128, true);
        b.set(199, true);
        b.set(200, true);

        assert_eq!((a.bitand(&b)).iter_ones().collect::<Vec<_>>(), vec![128, 200]);
        assert_eq!((a.bitor(&b)).iter_ones().collect::<Vec<_>>(), vec![127, 128, 199, 200]);
    }

    #[test]
    fn boundaries_and_zero_capacity() {
        let b0 = BitSet32::new(0);
        assert_eq!(b0.count_ones(), 0);
        assert_eq!((!b0).count_ones(), 0);

        let mut b1 = BitSet32::new(1);
        assert_eq!((!b1).iter_ones().collect::<Vec<_>>(), vec![0]);
        b1.set(0, true);
        assert_eq!((!b1).count_ones(), 0);

        assert_eq!((!BitSet32::new(32)).count_ones(), 32);
        assert_eq!((!BitSet64::new(64)).count_ones(), 64);
        assert_eq!((!BitSet128::new(128)).count_ones(), 128);

        let bv0 = BitSetVec::<1>::new(0);
        assert_eq!(bv0.count_ones(), 0);
        assert_eq!(bv0.not().count_ones(), 0);
    }

    #[test]
    #[should_panic]
    fn bitsetword_overflow_panics() {
        let _ = BitSetWord::<u32>::new(33);
    }

    #[test]
    #[should_panic]
    fn bitsetvec_overflow_panics() {
        let _ = BitSetVec::<1>::new(129);
    }
}
