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

#[cfg(test)]
mod tests {
    use super::*;
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
