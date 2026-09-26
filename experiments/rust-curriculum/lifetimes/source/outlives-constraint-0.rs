fn shorten<'a, 'b>(value: &'a str, _: &'b ()) -> &'b str { value }
fn main() { let _ = shorten("ore", &()); }
