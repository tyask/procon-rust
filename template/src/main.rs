use std::*;
use proconio::{input, fastout};
use fumin::*;

#[fastout]
fn main() {
}

#[allow(dead_code, unused_macros)]
mod fumin {
use std::*;
trait Joiner { fn join(self) -> String; }
pub trait Fmtx<T=Self> { fn fmtx(&self) -> String; }
macro_rules! impl_fmtx {
    ($($t:ty),*) => { $(impl Fmtx for $t { fn fmtx(&self) -> String { self.to_string() }})* }
}
#[macro_export]
macro_rules! out {
    ($($a:expr),*) => {
        let mut v = Vec::<String>::new();
        $(v.push(($a).fmtx());)*
        println!("{}", v.fmtx());
    };
}

impl<It: Iterator<Item=String>> Joiner for It { fn join(self) -> String { self.collect::<Vec<_>>().join(" ") } }
impl_fmtx!(i32, i64, &'static str, String);
impl<T: Fmtx> Fmtx for Vec<T> {
    fn fmtx(&self) -> String { self.iter().map(|e| e.fmtx()).join() }
}
impl<K: fmt::Display, V: Fmtx> Fmtx for collections::HashMap<K, V> {
    fn fmtx(&self) -> String { self.iter().map(|(k,v)| format!("{}:{}", k, v.fmtx())).join() }
}

}