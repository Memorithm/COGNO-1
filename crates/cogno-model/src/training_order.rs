//! Deterministic bounded epoch ordering for already selected training rows.
//! This utility has no access to labels, validation data or evaluation outcomes.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderError {
    Empty,
    Capacity,
    SamplingLimit,
}

/// Fisher-Yates permutation of 0..count using a specified SplitMix64 stream.
/// At most 4,096 rows and 16 draws per index; rejection avoids modulo bias.
/// Reproducibility utility only, not a cryptographic random generator.
pub fn epoch_order(count: usize, seed: u64, epoch: u64) -> Result<Vec<usize>, OrderError> {
    if count == 0 {
        return Err(OrderError::Empty);
    }
    if count > 4096 {
        return Err(OrderError::Capacity);
    }
    let mut state = seed.wrapping_add(epoch.wrapping_mul(0xd1b54a32d192ed03));
    let mut order: Vec<_> = (0..count).collect();
    for i in (1..count).rev() {
        let bound = (i + 1) as u64;
        let threshold = bound.wrapping_neg() % bound;
        let mut selected = None;
        for _ in 0..16 {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            if z >= threshold {
                selected = Some((z % bound) as usize);
                break;
            }
        }
        order.swap(i, selected.ok_or(OrderError::SamplingLimit)?);
    }
    Ok(order)
}

/// Errors for binary-label interleaving, distinct from the label-free shuffler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterleavedOrderError {
    Order(OrderError),
    InvalidLabel { index: usize },
    MissingClass,
}

/// Shuffle training indices once, then alternate classes while both remain.
/// Remaining majority rows form the tail. Every index occurs exactly once:
/// there is no oversampling, weighting or claim of class balance for unequal counts.
/// Inputs must contain only training labels and both binary classes.
pub fn binary_interleaved_epoch_order(
    labels: &[usize],
    seed: u64,
    epoch: u64,
) -> Result<Vec<usize>, InterleavedOrderError> {
    let order = epoch_order(labels.len(), seed, epoch).map_err(InterleavedOrderError::Order)?;
    let mut classes = [Vec::new(), Vec::new()];
    for index in order {
        let class = labels[index];
        if class > 1 {
            return Err(InterleavedOrderError::InvalidLabel { index });
        }
        classes[class].push(index);
    }
    if classes.iter().any(Vec::is_empty) {
        return Err(InterleavedOrderError::MissingClass);
    }
    let first = (seed.wrapping_add(epoch) & 1) as usize;
    let mut result = Vec::with_capacity(labels.len());
    for position in 0..classes[0].len().max(classes[1].len()) {
        for class in [first, 1 - first] {
            if let Some(&index) = classes[class].get(position) {
                result.push(index);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interleaving_preserves_every_row_and_alternates_without_oversampling() {
        for labels in [vec![0, 0, 0, 1, 1, 1], vec![0, 0, 0, 0, 1, 1], vec![0, 1]] {
            let order = binary_interleaved_epoch_order(&labels, 42, 7).unwrap();
            assert_eq!(
                order,
                binary_interleaved_epoch_order(&labels, 42, 7).unwrap()
            );
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..labels.len()).collect::<Vec<_>>());
            let minority = labels
                .iter()
                .filter(|&&label| label == 1)
                .count()
                .min(labels.iter().filter(|&&label| label == 0).count());
            for pair in order[..minority * 2].chunks_exact(2) {
                assert_ne!(labels[pair[0]], labels[pair[1]]);
            }
        }
    }
    #[test]
    fn interleaving_rejects_invalid_classes_and_capacity() {
        assert_eq!(
            binary_interleaved_epoch_order(&[], 0, 0),
            Err(InterleavedOrderError::Order(OrderError::Empty))
        );
        assert_eq!(
            binary_interleaved_epoch_order(&[0, 0], 0, 0),
            Err(InterleavedOrderError::MissingClass)
        );
        assert!(matches!(
            binary_interleaved_epoch_order(&[0, 2, 1], 0, 0),
            Err(InterleavedOrderError::InvalidLabel { index: 1 })
        ));
        assert_eq!(
            binary_interleaved_epoch_order(&vec![0; 4097], 0, 0),
            Err(InterleavedOrderError::Order(OrderError::Capacity))
        );
    }

    #[test]
    fn each_training_row_exactly_once_and_reproducible() {
        for count in [1, 2, 16, 4096] {
            for epoch in [0, 1, 23] {
                let a = epoch_order(count, 42, epoch).unwrap();
                assert_eq!(a, epoch_order(count, 42, epoch).unwrap());
                let mut sorted = a;
                sorted.sort_unstable();
                assert_eq!(sorted, (0..count).collect::<Vec<_>>());
            }
        }
        assert_ne!(epoch_order(32, 1, 0), epoch_order(32, 1, 1));
        assert_ne!(epoch_order(32, 1, 0), epoch_order(32, 7, 0));
    }
    #[test]
    fn bounds_and_zero_seed_are_explicit() {
        assert_eq!(epoch_order(0, 0, 0), Err(OrderError::Empty));
        assert_eq!(epoch_order(4097, 0, 0), Err(OrderError::Capacity));
        assert_eq!(epoch_order(1, 0, u64::MAX).unwrap(), vec![0]);
        assert!(epoch_order(16, 0, u64::MAX).is_ok());
    }
}
