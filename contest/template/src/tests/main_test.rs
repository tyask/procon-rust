
#[cfg(test)]
mod tests {
    use crate::{fumin::*, *};

    #[test]
    fn test_fmtx() {
        assert_eq!(fmtx!(2),           "2");
        assert_eq!(fmtx!(2, 3),        "2 3");
        assert_eq!(fmtx!(2.123),       "2.123");
        assert_eq!(fmtx!(vec![1,2,3]), "1 2 3");

        assert_eq!(fmtx!(2, 3;line),        "2\n3");

        assert_eq!(fmtx!(vec![1,2,3];byline),          "1\n2\n3");
        assert_eq!(fmtx!(vec!["ab","cd","ef"];byline), "ab\ncd\nef");

        assert_eq!(fmtx!(vec![1,2,3];debug), "[1, 2, 3]");
    }

    #[test]
    fn test_out() {
        out!(2);
        out!(2, 3);
        out!(2.123);
        out!(vec![1,2,3]);

        out!(vec![1,2,3];byline);
        out!(vec!["ab","cd","ef"];byline);

        out!(vec![1,2,3];debug);
    }

    #[test]
    fn test_chmax_chmin() {
        {
            let mut m = 0;
            let mut do_chmax = |v, exp_updated, exp_val| {
                assert_eq!(chmax(&mut m, &v), exp_updated);
                assert_eq!(m, exp_val);
            };
            do_chmax(1, true,  1);
            do_chmax(1, false, 1);
            do_chmax(0, false, 1);
        }

        {
            let mut m = 1;
            let mut do_chmin = |v, exp_updated, exp_val| {
                assert_eq!(chmin(&mut m, &v), exp_updated);
                assert_eq!(m, exp_val);
            };

            do_chmin(0, true,  0);
            do_chmin(0, false, 0);
            do_chmin(1, false, 0);
        }
    }
}