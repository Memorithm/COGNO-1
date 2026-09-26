macro_rules! inner{()=>{true}}macro_rules! outer{()=>{inner!()}}fn main(){let _:bool=outer!();}
