#![allow(dead_code)]

#[macro_export]
macro_rules! count {
     () => (0usize);
     ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

#[macro_export]
macro_rules! enrich_enum {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($e:ident,)* }) => {
        static_assertions::const_assert!(count!($($e,)*) <= 32);

        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, hash::Hash)]
        $(#[$meta])*
        $vis enum $name {
            $($e,)*
        }
        impl $name {
            pub const VALS: [Self; count!($($e)*)] = [$(Self::$e,)*];
            #[inline] pub const fn id(self) -> us { self as us }
            #[inline] pub const fn b(self) -> u32 { 1<<self.id() }
            #[inline] pub const fn is_or(self, b: u32) -> bool { self.b() & b > 0 }
            #[inline] pub const fn from_id(id: us) -> Self { Self::VALS[id] }
            #[inline] pub const fn from_bit(b: u32) -> Self {
                assert!(b.count_ones() <= 1);
                Self::from_id(b.trailing_zeros() as usize + 1)
            }
        }
        impl BitOr for $name {
            type Output = u32;
            fn bitor(self, rhs: Self) -> Self::Output { self.b() | rhs.b() }
        }

        impl From<us>  for $name { fn from(value: us) -> Self { Self::from_id(value) } }
        impl From<u32> for $name { fn from(value: u32) -> Self { Self::from_bit(value) } }
    };
}
