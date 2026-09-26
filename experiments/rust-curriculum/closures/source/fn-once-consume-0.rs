fn run(mut f:impl FnMut()){f()} fn main(){let s=String::from("closure");run(||drop(s));}
