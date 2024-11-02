#![allow(dead_code)]
use std::*;

pub struct Additive;
impl ac_library::Monoid for Additive {
    type S = (i64, i64);

    // e=identity()として、全てのaに対して
    // binary_operation(a,e)=binary_operation(e,a)=aとなるeを返す
    fn identity() -> Self::S { (0, 0) }

    // 区間取得を行うための演算
    fn binary_operation(a: &Self::S, b: &Self::S) -> Self::S { (a.0 + b.0, a.1 + b.1) }
}

// S: 各要素の型
pub struct LazyAdditive;
impl ac_library::lazysegtree::MapMonoid for LazyAdditive {
    type M = Additive;
    type F = i64; // F: Sに対する操作のための型. 遅延評価されている値がこの型で保持される.

    // id=identity_map()として、全てのaに対してmapping(id, a)=aとなるidを返す
    fn identity_map() -> Self::F { 0 }

    // 遅延評価値を各要素に反映させるための演算
    fn mapping(&f: &Self::F, &x: &(i64,i64)) -> (i64,i64) { (x.0 + f * x.1, x.1) }

    // 遅延している値に対して新しい操作を追加する演算
    // gが現在の操作、fが新しい操作
    fn composition(&f: &Self::F, &g: &Self::F) -> Self::F { g + f }
}
