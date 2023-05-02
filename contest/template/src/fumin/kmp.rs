#![allow(dead_code)]
use crate::{common::{is, ToUs}, or};

pub struct Kmp<T> {
    table: Vec<is>,
    s: Vec<T>,
}

impl<T: PartialEq+Clone> Kmp<T> {
    pub fn new(s: &[T]) -> Self {
        let mut t = vec![0_isize; s.len() + 1];
        t[0] = -1;
        let mut j = -1_isize;
        for i in 0..s.len() {
            while j >= 0 && s[i] != s[j.us()] { j = t[j.us()]; }
            j += 1;
            t[i+1] = or!(s[i+1] == s[j.us()]; t[j.us()], j);
        }
        Self { table: t, s: s.to_vec() }
    }

    // TODO 検索,周期

}