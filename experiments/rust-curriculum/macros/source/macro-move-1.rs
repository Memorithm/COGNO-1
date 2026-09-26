macro_rules! pair{($v:expr)=>{($v.clone(),$v)}}fn main(){let s=String::from("macro");let _=pair!(s);}
