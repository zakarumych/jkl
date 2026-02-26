use std::convert::Infallible;

#[inline(always)]
pub fn always_ok<T>(value: T) -> Result<T, Infallible> {
    Ok(value)
}

pub trait ResultExt<T> {
    fn unwrap_infallible(self) -> T;
}

impl<T> ResultExt<T> for Result<T, Infallible> {
    #[inline(always)]
    fn unwrap_infallible(self) -> T {
        match self {
            Ok(value) => value,
            Err(err) => match err {},
        }
    }
}
