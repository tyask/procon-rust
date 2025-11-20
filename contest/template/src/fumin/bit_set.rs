#![allow(dead_code)]
use std::hash;
use crate::common::*;

 trait BitSetTrait: Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + hash::Hash {
    fn with_capacity(n:us) -> Self;
    fn bitand(&self, other: &Self) -> Self;
    fn bitor(&self, other: &Self) -> Self;
    fn rev(&self) -> Self;
    fn count_ones(&self) -> us;
    fn ones(&self) -> Vec<us>;
    fn bitand_assign(&mut self, other: &Self);
    fn bitor_assign(&mut self, other: &Self);
    fn set(&mut self, i: us, f: bool);
}

#[macro_export] macro_rules! impl_bitset {
    ($t:ty, $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
        pub struct $name { b: $t, mask: $t }
        impl BitSetTrait for $name {
            fn with_capacity(n:us) -> Self {
                const SZ: us = std::mem::size_of::<$t>()*8;
                assert!(n <= SZ);
                let mask = if n == SZ { !0 } else { ((1 as $t)<<n)-1 };
                Self { b: 0, mask, }
            }
            fn bitand(&self, other: &Self) -> Self { Self { b: self.b & other.b, mask: self.mask } }
            fn bitor(&self, other: &Self) -> Self { Self { b: self.b | other.b, mask: self.mask } }
            fn rev(&self) -> Self { Self { b: (!self.b) & self.mask, mask: self.mask } }
            fn count_ones(&self) -> us { self.b.count_ones() as us }
            fn ones(&self) -> Vec<us> {
                let mut b = self.b;
                let mut ret = vec![];
                while b != 0 {
                    ret.push(b.trailing_zeros().us());
                    b &= b - 1;
                }
                ret
            }
            fn bitand_assign(&mut self, other: &Self) { self.b &= other.b }
            fn bitor_assign(&mut self, other: &Self) { self.b |= other.b; }
            fn set(&mut self, i: us, f: bool) {
                if f { self.b |= 1<<i; } else { self.b &= !(1<<i) }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:b}", self.b)
            }
        }
    }
}
impl_bitset!{u32,  BitSet32}
impl_bitset!{u64,  BitSet64}
impl_bitset!{u128, BitSet128}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
pub struct BitSetVec<const BLOCK: us> {
    b: [BitSet128; BLOCK],
    block: us,
}
impl<const BLOCK: us> BitSetVec<BLOCK> {
    const M: us = std::mem::size_of::<u128>() * 8;
}
impl<const BLOCK: us> BitSetTrait for BitSetVec<BLOCK> {
    fn with_capacity(n:us) -> Self {
        let mut b = [BitSet128::with_capacity(Self::M); BLOCK];
        let block = (n.saturating_sub(1)>>7)+1;
        b[block-1] = BitSet128::with_capacity(n&(Self::M-1));
        b[block..].fill(BitSet128::with_capacity(0));
        Self {
            b, 
            block,
        }
    }
    fn bitand(&self, other: &Self) -> Self {
        let mut t = self.clone();
        t.bitand_assign(other);
        t
    }
    fn bitor(&self, other: &Self) -> Self {
        let mut t = self.clone();
        t.bitor_assign(other);
        t
    }
    fn rev(&self) -> Self {
        let mut t = self.clone();
        for i in 0..self.block { t.b[i] = self.b[i].rev(); }
        t
    }
    fn count_ones(&self) -> us { self.b.iter().map(|b|b.count_ones()).sum::<us>() }
    fn ones(&self) -> Vec<us> {
        (0..self.block).flat_map(|i|self.b[i].ones().into_iter().map(move|bi|i*Self::M+bi)).cv()
    }
    fn bitand_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i].bitand_assign(&other.b[i]); }
    }
    fn bitor_assign(&mut self, other: &Self) {
        for i in 0..self.block { self.b[i].bitor_assign(&other.b[i]); }
    }
    fn set(&mut self, i: us, f: bool) {
        self.b[i>>7].set(i&(Self::M-1), f);
    }
}
