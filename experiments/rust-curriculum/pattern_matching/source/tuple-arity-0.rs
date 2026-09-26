enum Pair { Values(i32,i32) } fn main(){ let Pair::Values(a)=Pair::Values(2,4); let _=a; }
