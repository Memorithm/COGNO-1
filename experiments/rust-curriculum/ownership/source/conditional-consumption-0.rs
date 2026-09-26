fn main() { let item = String::from("ore"); let choose = std::env::args().len() > 1; if choose { drop(item); } let _ = item.len(); }
