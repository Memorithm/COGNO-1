fn run(f:impl FnOnce()){f()} fn main(){let s=String::from("closure");run(||drop(s));}
