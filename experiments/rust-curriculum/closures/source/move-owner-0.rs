fn main(){let s=String::from("owner");let f=move||s.len();drop(s);let _=f();}
