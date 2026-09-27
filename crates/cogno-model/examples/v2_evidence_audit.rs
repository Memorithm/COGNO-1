#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
fn main() -> Result<(), String> {
    let e = load(&std::env::args().skip(1).collect::<Vec<_>>())?;
    header(&e);
    println!("seeds,unique_sources,observations");
    println!(
        "{},{},{}",
        e.seeds.len(),
        e.rows.len() / e.seeds.len(),
        e.rows.len()
    );
    Ok(())
}
