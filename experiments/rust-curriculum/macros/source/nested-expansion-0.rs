macro_rules! inner{()=>{1u8}}macro_rules! outer{()=>{inner!()}}fn main(){let _:bool=outer!();}
