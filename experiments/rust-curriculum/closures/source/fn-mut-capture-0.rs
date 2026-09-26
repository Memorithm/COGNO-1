fn apply(f:impl Fn()){f()} fn main(){let mut n=0;apply(||n+=1);}
