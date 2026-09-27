//! Explicitly budgeted, hash-bound research training; no model promotion.
#![forbid(unsafe_code)]
mod train_v2_support;
use std::path::Path;
use train_v2_support::*;
const USAGE: &str = "rust_train_v2 COMMAND PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE [paths]\ncommands: plan; train NEW_OUTPUT; verify OUTPUT COMPLETE_SHA; select OUTPUT COMPLETE_SHA NEW_SELECTION; test OUTPUT COMPLETE_SHA SELECTION NEW_TEST_OUTPUT; resume PRIOR_OUTPUT NEW_OUTPUT";
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") => {
            let a = admit(&args[1..])?;
            print!("{}", plan(&a));
        }
        Some("train") if args.len() == 6 => {
            let a = admit(&args[1..5])?;
            train(&a, Path::new(&args[5]), None)?;
        }
        Some("verify") if args.len() == 7 => {
            let a = admit(&args[1..5])?;
            verify(&a, Path::new(&args[5]), &args[6])?;
            println!("bundle verified");
        }
        Some("select") if args.len() == 8 => {
            let a = admit(&args[1..5])?;
            select(&a, Path::new(&args[5]), &args[6], Path::new(&args[7]))?;
        }
        Some("test") if args.len() == 9 => {
            let a = admit(&args[1..5])?;
            test_selected(
                &a,
                Path::new(&args[5]),
                &args[6],
                Path::new(&args[7]),
                Path::new(&args[8]),
            )?;
        }
        Some("resume") if args.len() == 7 => {
            let a = admit(&args[1..5])?;
            train(&a, Path::new(&args[6]), Some(Path::new(&args[5])))?;
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}
