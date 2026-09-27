//! Leave-one-group diagnostics reveal aggregate deltas dominated by one domain.
#![forbid(unsafe_code)]
mod evaluation_v2_support;
use evaluation_v2_support::*;
use std::collections::BTreeMap;
type Deltas = BTreeMap<String, Vec<(f64, f64)>>;
fn aggregate(groups: &Deltas, omit: Option<&str>) -> Result<(usize, usize, [f64; 4]), String> {
    let retained: Vec<_> = groups
        .iter()
        .filter(|(g, _)| Some(g.as_str()) != omit)
        .collect();
    if retained.is_empty() {
        return Err("no retained groups".into());
    }
    let observations = retained.iter().map(|(_, r)| r.len()).sum::<usize>();
    let equal_accuracy = mean(retained.iter().map(|(_, r)| mean(r.iter().map(|x| x.0))));
    let equal_nll = mean(retained.iter().map(|(_, r)| mean(r.iter().map(|x| x.1))));
    let weighted_accuracy = mean(retained.iter().flat_map(|(_, r)| r.iter().map(|x| x.0)));
    let weighted_nll = mean(retained.iter().flat_map(|(_, r)| r.iter().map(|x| x.1)));
    Ok((
        retained.len(),
        observations,
        [equal_accuracy, equal_nll, weighted_accuracy, weighted_nll],
    ))
}
fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 9 {
        return Err(format!("usage: {USAGE} OTHER_PREDICTIONS SHA SPLIT"));
    }
    let first = load(&a[..6])?;
    let mut other = a[..6].to_vec();
    other[0] = a[6].clone();
    other[1] = a[7].clone();
    let second = load(&other)?;
    let rows = paired(&first, &second, &a[8])?;
    let mut groups: Deltas = BTreeMap::new();
    for (x, y) in rows {
        groups.entry(x.group.clone()).or_default().push((
            f64::from(y.prediction == y.target) - f64::from(x.prediction == x.target),
            nll(y.target, y.probability) - nll(x.target, x.probability),
        ));
    }
    if groups.len() < 2 {
        return Err("need at least two groups for omission sensitivity".into());
    }
    header(&first);
    println!("# other_arm={} other_predictions_sha256={}; descriptive_omission_not_retraining_or_confidence_interval",second.arm,second.bindings[0]);
    println!("split,scope,omitted_group,retained_groups,unique_sources,seeds,equal_group_accuracy_delta,equal_group_nll_delta,source_weighted_accuracy_delta,source_weighted_nll_delta");
    for omit in std::iter::once(None).chain(groups.keys().map(|g| Some(g.as_str()))) {
        let (n, count, m) = aggregate(&groups, omit)?;
        println!(
            "{},{},{},{n},{},{},{:.12},{:.12},{:.12},{:.12}",
            a[8],
            if omit.is_some() {
                "leave_one_out"
            } else {
                "full"
            },
            omit.unwrap_or("NONE"),
            count / first.seeds.len(),
            first.seeds.len(),
            m[0],
            m[1],
            m[2],
            m[3]
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinguishes_group_and_source_weighting_and_sign_reversal() {
        let g = BTreeMap::from([
            ("small".into(), vec![(1.0, -1.0)]),
            ("large".into(), vec![(-0.5, 0.5); 3]),
        ]);
        let full = aggregate(&g, None).unwrap();
        assert_eq!(full.2, [0.25, -0.25, -0.125, 0.125]);
        assert_eq!(
            aggregate(&g, Some("large")).unwrap().2,
            [1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(
            aggregate(&g, Some("small")).unwrap().2,
            [-0.5, 0.5, -0.5, 0.5]
        );
        assert!(aggregate(&BTreeMap::new(), None).is_err());
    }
}
