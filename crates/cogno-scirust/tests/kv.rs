//! KV cache tests (COGNO-1 §17, §SciRust #8).

use cogno_scirust::kv::{BoundedKvCache, KvCachePolicy, KvPushOutcome};

#[test]
fn kv_reject_on_overflow() {
    let mut kv = BoundedKvCache::try_new(2, 4, 2, 1, KvCachePolicy::RejectOnOverflow).unwrap();
    assert_eq!(kv.push(1), KvPushOutcome::Admitted);
    assert_eq!(kv.push(2), KvPushOutcome::Admitted);
    assert_eq!(kv.push(3), KvPushOutcome::Refused);
}

#[test]
fn kv_sliding_window_evicts_oldest() {
    let mut kv = BoundedKvCache::try_new(
        4,
        4,
        2,
        1,
        KvCachePolicy::SlidingWindow { window_tokens: 2 },
    )
    .unwrap();
    kv.push(1);
    kv.push(2);
    let out = kv.push(3);
    assert_eq!(
        out,
        KvPushOutcome::Evicted {
            dropped_token_id: 1
        }
    );
    assert_eq!(kv.len, 2);
    assert_eq!(kv.token_ids[0], 2);
    assert_eq!(kv.token_ids[1], 3);
}

#[test]
fn kv_prefix_pinned_sliding_window() {
    let mut kv = BoundedKvCache::try_new(
        6,
        4,
        2,
        1,
        KvCachePolicy::PrefixPinnedSlidingWindow {
            prefix_tokens: 2,
            window_tokens: 2,
        },
    )
    .unwrap();
    kv.push(1);
    kv.push(2);
    kv.push(3);
    kv.push(4);
    let out = kv.push(5);
    assert_eq!(
        out,
        KvPushOutcome::Evicted {
            dropped_token_id: 3
        }
    );
    assert_eq!(kv.len, 4);
    assert_eq!(kv.token_ids[0], 1);
    assert_eq!(kv.token_ids[1], 2);
    assert_eq!(kv.token_ids[2], 4);
    assert_eq!(kv.token_ids[3], 5);
}

#[test]
fn kv_report_never_silent() {
    let kv = BoundedKvCache::try_new(
        4,
        4,
        2,
        1,
        KvCachePolicy::SlidingWindow { window_tokens: 2 },
    )
    .unwrap();
    let r = kv.report(10);
    assert_eq!(r.requested_tokens, 10);
    assert_eq!(r.admitted_tokens, 0);
    assert_eq!(r.dropped_tokens, 10);
}
