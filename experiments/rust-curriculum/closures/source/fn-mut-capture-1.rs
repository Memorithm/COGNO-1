fn apply(mut f:impl FnMut()){f()} fn main(){let mut n=0;apply(||n+=1);}
