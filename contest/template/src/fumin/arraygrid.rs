#![allow(dead_code)]
use std::ops::{Index, IndexMut};

use crate::common::*;
use super::pt::Pt;

// CAP(fumin::pt)

#[derive(Debug, Clone)]
pub struct ArrayGrid<const H: us, const W: us, const N: us, T> {
    g: [T; N],
}

impl<const H: us, const W: us, const N: us, T> ArrayGrid<H, W, N, T> {
    #[inline]
    const fn idx(i: us, j: us) -> us {
        i * W + j
    }

    #[inline]
    fn assert_shape() {
        assert!(N == H * W, "ArrayGrid shape mismatch: N({}) != H({})*W({})", N, H, W);
    }

    pub fn is_in_p<M: IntoT<us>>(&self, p: Pt<M>) -> bool {
        self.is_in_t(p.tuple())
    }

    pub fn is_in_t<M: IntoT<us>>(&self, t: (M, M)) -> bool {
        t.0.into_t() < H && t.1.into_t() < W
    }
}

impl<const H: us, const W: us, const N: us, T: Copy> ArrayGrid<H, W, N, T> {
    pub fn with_default(v: T) -> Self {
        Self::assert_shape();
        Self { g: [v; N] }
    }
}

impl<const H: us, const W: us, const N: us, T: Copy + Default> ArrayGrid<H, W, N, T> {
    pub fn new() -> Self {
        Self::assert_shape();
        Self { g: [T::default(); N] }
    }
}

impl<const H: us, const W: us, const N: us> ToString for ArrayGrid<H, W, N, char> {
    fn to_string(&self) -> String {
        let mut ret = String::new();
        for i in 0..H {
            ret.push_str(format!("{}: ", i % 10).as_str()); // line number
            ret.push_str(self[i].str().as_str());
            ret.push('\n');
        }
        ret
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> Index<I> for ArrayGrid<H, W, N, T> {
    type Output = [T];

    fn index(&self, i: I) -> &Self::Output {
        let row = i.into_t();
        let idx = row * W;
        &self.g[idx..idx + W]
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> IndexMut<I>
    for ArrayGrid<H, W, N, T>
{
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        let row = i.into_t();
        let idx = row * W;
        &mut self.g[idx..idx + W]
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> Index<(I, I)>
    for ArrayGrid<H, W, N, T>
{
    type Output = T;

    fn index(&self, index: (I, I)) -> &Self::Output {
        &self.g[Self::idx(index.0.into_t(), index.1.into_t())]
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> IndexMut<(I, I)>
    for ArrayGrid<H, W, N, T>
{
    fn index_mut(&mut self, index: (I, I)) -> &mut Self::Output {
        &mut self.g[Self::idx(index.0.into_t(), index.1.into_t())]
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> Index<Pt<I>> for ArrayGrid<H, W, N, T> {
    type Output = T;

    fn index(&self, p: Pt<I>) -> &Self::Output {
        &self[p.tuple()]
    }
}

impl<const H: us, const W: us, const N: us, T, I: IntoT<us>> IndexMut<Pt<I>>
    for ArrayGrid<H, W, N, T>
{
    fn index_mut(&mut self, p: Pt<I>) -> &mut Self::Output {
        &mut self[p.tuple()]
    }
}

impl<const H: us, const W: us, const N: us, T: Clone> From<&Vec<Vec<T>>> for ArrayGrid<H, W, N, T> {
    fn from(value: &Vec<Vec<T>>) -> Self {
        Self::assert_shape();
        assert!(value.len() == H, "ArrayGrid height mismatch: got {}, expected {}", value.len(), H);
        assert!(value.iter().all(|row| row.len() == W), "ArrayGrid width mismatch");

        let g = std::array::from_fn(|idx| {
            let i = idx / W;
            let j = idx % W;
            value[i][j].clone()
        });
        Self { g }
    }
}
