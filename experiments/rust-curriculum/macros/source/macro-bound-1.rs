macro_rules! make{($t:ty)=>{fn dup(v:$t)->$t{v.clone()}}}make!(String);fn main(){}
