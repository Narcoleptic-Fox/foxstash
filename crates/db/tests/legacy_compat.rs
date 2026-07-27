//! Opens a collection written by the PUBLISHED foxstash 0.6.0.
//!
//! The migration test in core builds its "legacy" checkpoint by stripping fields
//! from one this build wrote — a synthetic v0. That proves the code path, not the
//! compatibility. This opens a directory produced by the real 0.6.0 crate from
//! crates.io, which is what an actual user has on disk.
//!
//! Skipped unless FOXSTASH_LEGACY_COLLECTION points at such a directory; see
//! the KB note for how to regenerate one.

use foxstash_db::{Collection, DbConfig};

fn vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5
        })
        .collect()
}

#[test]
fn a_collection_written_by_published_0_6_0_opens_and_migrates() {
    let Ok(src) = std::env::var("FOXSTASH_LEGACY_COLLECTION") else {
        eprintln!("skipping: set FOXSTASH_LEGACY_COLLECTION to a 0.6.0-written collection");
        return;
    };
    // Copy, so the test can run repeatedly against a pristine source.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("legacy");
    std::fs::create_dir_all(&path).unwrap();
    for e in std::fs::read_dir(&src).unwrap().flatten() {
        std::fs::copy(e.path(), path.join(e.file_name())).unwrap();
    }

    let meta_before = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .find(|e| e.path().extension().is_some_and(|x| x == "meta"))
        .expect("a 0.6.0 checkpoint meta");
    let raw = std::fs::read_to_string(meta_before.path()).unwrap();
    assert!(
        !raw.contains("format_version"),
        "the fixture should be genuinely pre-versioning; got {raw}"
    );

    let cfg = DbConfig::default().with_embedding_dim(16);
    let c = Collection::open("legacy", &path, cfg).expect(
        "a collection written by published 0.6.0 must open — this is the migration \
         path meeting real old data rather than a synthetic v0",
    );
    assert_eq!(
        c.len(),
        500,
        "every 0.6.0 document must survive the migration"
    );

    // The data must be usable, not merely countable.
    for i in [0usize, 250, 499] {
        assert_eq!(
            c.get(&format!("d{i}")).unwrap().map(|d| d.content),
            Some(format!("legacy content {i}")),
            "d{i} content must round-trip from 0.6.0"
        );
        let hits = c.search(&vector(i as u64, 16), 1, None).unwrap();
        assert_eq!(hits[0].id, format!("d{i}"), "d{i} must still be searchable");
    }

    // And it must still be writable afterwards.
    c.insert(
        "new".into(),
        "after migration".into(),
        vector(9999, 16),
        None,
    )
    .unwrap();
    assert_eq!(c.len(), 501);

    // The meta should have been stamped, so the migration happens once.
    let raw_after = std::fs::read_to_string(meta_before.path()).unwrap();
    assert!(
        raw_after.contains("format_version"),
        "the checkpoint meta should have been migrated in place"
    );
}
