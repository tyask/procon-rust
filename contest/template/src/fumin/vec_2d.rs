#![allow(dead_code)]
use crate::common::*;

trait Vec2d<T> {
    fn rot_right(&self) -> Vec<Vec<T>>;
    fn rot_left(&self) -> Vec<Vec<T>>;
}

impl<T: Clone> Vec2d<T> for [Vec<T>] {
    // 2次元配列を右に90度回転させる
    fn rot_right(&self) -> Vec<Vec<T>> {
        let (h, w) = (self.len(), self[0].len());
        let mut a = vec![vec![self[0][0].clone(); h]; w];
        for i in 0..h { for j in 0..w { a[j][h-1-i]=self[i][j].clone(); }}
        a
    }

    // 2次元配列を左に90度回転させる
    fn rot_left(&self) -> Vec<Vec<T>> {
        let (h, w) = (self.len(), self[0].len());
        let mut a = vec![vec![self[0][0].clone(); h]; w];
        for i in 0..h { for j in 0..w { a[w-j-1][i]=self[i][j].clone(); }}
        a
    }
}