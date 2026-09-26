macro_rules! make{($t:ty)=>{fn dup(v:$t)->$t{v.clone()}}}struct NoClone;make!(NoClone);fn main(){}
