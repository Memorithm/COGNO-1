fn main(){ let input=Some(String::from("pattern")); if let Some(ref s)=input { let _=s.len(); } drop(input); }
