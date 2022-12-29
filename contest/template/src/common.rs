
pub trait From<T> { fn from(t: T) -> Self; }

pub trait Unit {
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const TEN: Self;
}
