//! Paired source/seed deltas are reduced to groups before bootstrap sampling.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::BTreeMap;
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn bootstrap(deltas: &[f64], mut state: u64) -> Result<(f64, f64, f64), String> {
    if deltas.len() < 2 || deltas.len() > 4096 || deltas.iter().any(|x| !x.is_finite()) {
        return Err("need 2..4096 finite group deltas".into());
    }
    let bound = deltas.len() as u64;
    let threshold = bound.wrapping_neg() % bound;
    let mut samples = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sum = 0.0;
        for _ in deltas {
            let mut selected = None;
            for _ in 0..16 {
                let z = next(&mut state);
                if z >= threshold {
                    selected = Some((z % bound) as usize);
                    break;
                }
            }
            sum += deltas[selected.ok_or("sampling bound")?];
        }
        samples.push(sum / deltas.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok((mean(deltas.iter().copied()), samples[49], samples[1949]))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 10 {
        return Err(format!(
            "usage: {USAGE} OTHER_PREDICTIONS SHA SPLIT BOOTSTRAP_SEED"
        ));
    }
    let first = load(&a[..6])?;
    let mut second_args = a[..6].to_vec();
    second_args[0] = a[6].clone();
    second_args[1] = a[7].clone();
    let second = load(&second_args)?;
    let rows = paired(&first, &second, &a[8])?;
    let seed = a[9].parse::<u64>().map_err(|e| e.to_string())?;
    let mut groups: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
    for (x, y) in &rows {
        groups.entry(&x.group).or_default().push((
            f64::from(y.prediction == y.target) - f64::from(x.prediction == x.target),
            nll(y.target, y.probability) - nll(x.target, x.probability),
        ));
    }
    let accuracy: Vec<_> = groups
        .values()
        .map(|g| mean(g.iter().map(|v| v.0)))
        .collect();
    let loss: Vec<_> = groups
        .values()
        .map(|g| mean(g.iter().map(|v| v.1)))
        .collect();
    let ac = bootstrap(&accuracy, seed)?;
    let nl = bootstrap(&loss, seed)?;
    header(&first);
    println!(
        "# other_arm={} other_predictions_sha256={}",
        second.arm, second.bindings[0]
    );
    println!("# exploratory_cluster_percentiles_assume_independent_groups; few_groups_unreliable; fixed_seeds_not_resampled");
    println!(
        "split,groups,unique_sources,seeds,resamples,bootstrap_seed,metric,b_minus_a,p025,p975"
    );
    for (metric, m) in [("accuracy", ac), ("nll", nl)] {
        println!(
            "{},{},{},{},2000,{seed},{metric},{:.12},{:.12},{:.12}",
            a[8],
            groups.len(),
            rows.len() / first.seeds.len(),
            first.seeds.len(),
            m.0,
            m.1,
            m.2
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_deterministic_cluster_distribution() {
        assert_eq!(bootstrap(&[1.0; 3], 42).unwrap(), (1.0, 1.0, 1.0));
        assert!(bootstrap(&[1.0], 42).is_err());
        let a = bootstrap(&[-1.0, 0.0, 0.5], 42).unwrap();
        assert_eq!(a, bootstrap(&[-1.0, 0.0, 0.5], 42).unwrap());
        assert!(a.1 >= -1.0 && a.2 <= 0.5);
    }
}
