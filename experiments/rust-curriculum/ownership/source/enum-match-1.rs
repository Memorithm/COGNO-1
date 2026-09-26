enum Item { Named(String) }
fn main() { let item = Item::Named("ore".into()); let Item::Named(ref name) = item; let _ = name.len(); let _saved = item; }
