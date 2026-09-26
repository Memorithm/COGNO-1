struct Parcel { name: String, code: String }
fn main() { let p = Parcel { name: "ore".into(), code: "A".into() }; let n = p.name; let _ = (p.code.len(), n); }
