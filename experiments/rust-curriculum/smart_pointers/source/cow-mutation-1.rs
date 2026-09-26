fn main(){let mut x:std::borrow::Cow<str>=std::borrow::Cow::Borrowed("cow");x.to_mut().push_str("s");}
