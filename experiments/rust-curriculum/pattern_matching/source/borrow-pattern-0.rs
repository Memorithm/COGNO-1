fn main(){ let input=Some(String::from("pattern")); if let Some(s)=input { let _=s.len(); } drop(input); }
