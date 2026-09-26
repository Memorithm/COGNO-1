fn read()->Result<u8,String>{let x=Err::<u8,u32>(9).map_err(|e|e.to_string())?;Ok(x)} fn main(){}
