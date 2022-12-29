
#[cfg(test)]
mod tests {
    use crate::{common::*, *};

    #[test]
    fn test_fmtx() {
        assert_eq!(fmt!(2),           "2");
        assert_eq!(fmt!(2, 3),        "2 3");
        assert_eq!(fmt!(2.123),       "2.123");
        assert_eq!(fmt!(vec![1,2,3]), "1 2 3");

        assert_eq!(fmt!(@line 2, 3),        "2\n3");

        assert_eq!(fmt!(@byline vec![1,2,3]),          "1\n2\n3");
        assert_eq!(fmt!(@byline vec!["ab","cd","ef"]), "ab\ncd\nef");

        assert_eq!(fmt!(@debug vec![1,2,3]), "[1, 2, 3]");
    }

    #[test]
    fn test_chmax_chmin() {
        {
            let mut m = 0;
            let mut do_chmax = |v, exp_updated, exp_val| {
                assert_eq!(chmax(v, &mut m), exp_updated);
                assert_eq!(m, exp_val);
            };
            do_chmax(&1, true,  1);
            do_chmax(&1, false, 1);
            do_chmax(&0, false, 1);
        }

        {
            let mut m = 1;
            let mut do_chmin = |v, exp_updated, exp_val| {
                assert_eq!(chmin(v, &mut m), exp_updated);
                assert_eq!(m, exp_val);
            };

            do_chmin(&0, true,  0);
            do_chmin(&0, false, 0);
            do_chmin(&1, false, 0);
        }
    }
}