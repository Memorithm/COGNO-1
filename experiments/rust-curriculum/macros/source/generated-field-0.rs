macro_rules! field{($v:expr,$n:ident)=>{$v.$n}}struct P{x:u8}fn main(){let _=field!(P{x:1},y);}
