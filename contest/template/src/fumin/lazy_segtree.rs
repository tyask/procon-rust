#![allow(dead_code)]
use std::*;

pub struct Additive<S>(convert::Infallible, marker::PhantomData<fn() -> S>);
impl<S> ac_library::Monoid for Additive<S>
where
    S: Copy + ops::Add<Output = S> + num_traits::Zero, {
    type S = S;

    // e=identity()として、全てのaに対して
    // binary_operation(a,e)=binary_operation(e,a)=aとなるeを返す
    fn identity() -> Self::S { S::zero() }

    // 区間取得を行うための演算
    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S { *a + *b }
}

// S: 各要素の型
pub struct LazyAdditive<S>(std::marker::PhantomData<fn()->S>);
impl<S> ac_library::lazysegtree::MapMonoid for LazyAdditive<S>
where
    S: Copy + ops::Add<Output=S> + num_traits::Zero {
    type M = Additive<S>;
    type F = S; // F: Sに対する操作のための型. 遅延評価されている値がこの型で保持される.

    // id=identity_map()として、全てのaに対してmapping(id, a)=aとなるidを返す
    fn identity_map() -> Self::F { S::zero() }

    // 遅延評価値を各要素に反映させるための演算
    fn mapping(&f: &Self::F, &x: &S) -> S { f + x }

    // 遅延している値に対して新しい操作を追加する演算
    // gが現在の操作、fが新しい操作
    fn composition(&f: &Self::F, &g: &Self::F) -> Self::F { g + f }
}
