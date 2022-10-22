
#[cfg(test)]
mod tests {
    use crate::fumin::*;

    #[test]
    fn test_fmtx() {
        assert_eq!(2.fmtx(),     "2");
        assert_eq!(2.123.fmtx(), "2.123");
        assert_eq!(vec![1,2,3].fmtx(), "1 2 3");
    }
}