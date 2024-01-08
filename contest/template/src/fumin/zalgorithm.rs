#![allow(dead_code)]

// sと任意のiについてs[i..]の最長共通接頭辞の長さを計算する.
pub fn zalgorithm<T: PartialEq>(s: &[T]) -> Vec<usize> {
    let (mut i, mut j, n) = (1, 0, s.len());
    let mut a = vec![0; s.len()];
    while i < n {
        while i + j < n && s[j] == s[i+j] { j += 1; }
        a[i] = j;
        if j == 0 { i += 1; continue; }
        let mut k = 1;
        while i + k < n && k + a[k] < j { a[i+k] = a[k]; k += 1; }
        i += k; j -= k;
    }
    a
}
