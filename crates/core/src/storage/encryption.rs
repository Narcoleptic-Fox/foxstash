//! Caller-supplied encryption at rest (issue #50).
//!
//! foxstash never holds a key and never runs a KDF. A consumer handling sensitive material (PHI,
//! privileged content) supplies a [`Cipher`] whose key it owns and should zeroize; foxstash runs
//! every byte it persists (WAL entries, checkpoints, the manifest, graph snapshots) through that
//! cipher when one is configured, and leaves them in the clear when one is not.
//!
//! # What foxstash owns vs. what the caller owns
//!
//! The caller's `Cipher` owns the key, the nonce, and the authentication tag. foxstash owns the
//! *framing*: a small plaintext header on each artifact, and the **associated data** (AAD) built
//! from it. The header binds every artifact to its [`Role`] (a WAL segment cannot be replayed as a
//! snapshot), to a per-store [`StoreId`] (an artifact from another store cannot be swapped in), and
//! to a `generation` (a stale checkpoint cannot be rolled back under a newer manifest). Because the
//! header is the AAD, altering any of it fails the open.
//!
//! # Fail closed
//!
//! Every read that cannot be authenticated is an error. There is no silent fallback (e.g. rebuild
//! from the WAL), because that would hide tampering or key loss behind a slow-but-successful open.
//!
//! # Forward compatibility
//!
//! Each artifact carries a one-byte framing version. The current framing is one-shot: the whole
//! artifact is sealed in a single call. A future chunked/streaming framing can bump the version
//! without a migration guess, because a reader rejects a version it does not understand.

/// Error returned by [`Cipher::open`] and the artifact readers here.
///
/// Deliberately a single opaque message: a cipher should not report *why* an open failed (wrong
/// key vs. bad tag vs. wrong AAD), since that can leak information to an attacker probing the store.
#[derive(Debug, Clone)]
pub struct CipherError(pub String);

impl std::fmt::Display for CipherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "decryption failed: {}", self.0)
    }
}

impl std::error::Error for CipherError {}

/// Authenticated encryption supplied by the caller.
///
/// Implementations must be an AEAD (e.g. XChaCha20-Poly1305): `open` must return an error unless
/// the ciphertext was produced by `seal` under the same key **and** the same `aad`. `aad` is
/// authenticated but not encrypted. foxstash constructs `aad`; the implementation
/// must bind it into the tag and must manage its own nonce (typically by prepending it to the
/// returned bytes). foxstash constructs `aad` from the artifact header (role, store id,
/// generation); see [`seal_artifact`].
pub trait Cipher: Send + Sync {
    /// Encrypt and authenticate `plaintext`, binding `aad`. The returned bytes are opaque to
    /// foxstash (they include whatever nonce/tag the scheme needs).
    fn seal(&self, plaintext: &[u8], aad: &[u8]) -> Vec<u8>;

    /// Authenticate against `aad` and decrypt. Returns an error on any key, tag, or `aad` mismatch,
    /// or on malformed input. Must not panic on attacker-controlled bytes.
    fn open(&self, ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CipherError>;
}

/// The kind of artifact an encrypted blob holds, bound into the AAD so one role cannot be passed
/// off as another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Role {
    Wal = 1,
    Checkpoint = 2,
    Snapshot = 3,
    Manifest = 4,
}

impl Role {
    fn from_byte(b: u8) -> Option<Role> {
        match b {
            1 => Some(Role::Wal),
            2 => Some(Role::Checkpoint),
            3 => Some(Role::Snapshot),
            4 => Some(Role::Manifest),
            _ => None,
        }
    }
}

/// A per-store identifier, generated at store creation and bound into every artifact's AAD so an
/// artifact from a different store (or a different install) cannot be substituted.
pub type StoreId = [u8; 16];

/// Magic prefix identifying a foxstash encrypted artifact ("FoXstash Encrypted").
const MAGIC: [u8; 4] = *b"FXEN";

/// Framing version. Bump only when the on-disk layout of an encrypted artifact changes; a reader
/// rejects a version it does not recognize rather than guessing.
pub const ENC_FORMAT_VERSION: u8 = 1;

/// Length of the plaintext header: magic(4) + version(1) + role(1) + store_id(16) + generation(8).
const HEADER_LEN: usize = 4 + 1 + 1 + 16 + 8;

/// Build the plaintext header / AAD for an artifact. Bytes 0..HEADER_LEN are written verbatim on
/// disk *and* passed to the cipher as AAD, so any tampering with role, store, or generation fails
/// the open.
fn header(role: Role, store_id: &StoreId, generation: u64) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(&MAGIC);
    h[4] = ENC_FORMAT_VERSION;
    h[5] = role as u8;
    h[6..22].copy_from_slice(store_id);
    h[22..30].copy_from_slice(&generation.to_le_bytes());
    h
}

/// Seal `plaintext` into an on-disk artifact: the plaintext header followed by the cipher's sealed
/// body. The header is the AAD.
pub fn seal_artifact(
    cipher: &dyn Cipher,
    role: Role,
    store_id: &StoreId,
    generation: u64,
    plaintext: &[u8],
) -> Vec<u8> {
    let hdr = header(role, store_id, generation);
    let body = cipher.seal(plaintext, &hdr);
    let mut out = Vec::with_capacity(hdr.len() + body.len());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&body);
    out
}

/// Authenticate and open an artifact sealed by [`seal_artifact`].
///
/// Verifies the magic, framing version, and that the artifact belongs to `expected_store`, then
/// authenticates the body against the header and decrypts. On success returns the authenticated
/// [`Role`] and `generation` (trustworthy, since the header is AAD) alongside the plaintext — the
/// caller asserts those are the role and generation it expected (rollback / role-confusion check).
///
/// Fail closed: a bad magic, unknown version, foreign store, or any authentication failure is an
/// error, never a partial or fallback result.
pub fn open_artifact(
    cipher: &dyn Cipher,
    expected_store: &StoreId,
    data: &[u8],
) -> Result<(Role, u64, Vec<u8>), CipherError> {
    if data.len() < HEADER_LEN {
        return Err(CipherError("artifact shorter than its header".into()));
    }
    let (hdr, body) = data.split_at(HEADER_LEN);
    if hdr[0..4] != MAGIC {
        return Err(CipherError("not a foxstash encrypted artifact".into()));
    }
    if hdr[4] != ENC_FORMAT_VERSION {
        return Err(CipherError(format!(
            "unsupported encryption framing version {} (this build writes {ENC_FORMAT_VERSION})",
            hdr[4]
        )));
    }
    let role = Role::from_byte(hdr[5])
        .ok_or_else(|| CipherError(format!("unknown artifact role {}", hdr[5])))?;
    if hdr[6..22] != expected_store[..] {
        return Err(CipherError("artifact belongs to a different store".into()));
    }
    let generation = u64::from_le_bytes(hdr[22..30].try_into().expect("8 bytes"));
    // AAD is the exact on-disk header, so a flipped role/store/generation byte fails here.
    let plaintext = cipher.open(body, hdr)?;
    Ok((role, generation, plaintext))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy authenticated cipher for exercising the framing contract. NOT real crypto: a real
    /// caller supplies XChaCha20-Poly1305 or similar. It detects a wrong key (keystream differs, so
    /// the recovered plaintext fails its tag) and a wrong AAD (the tag covers the AAD).
    struct ToyCipher {
        key: u8,
    }

    impl ToyCipher {
        fn tag(&self, aad: &[u8], plaintext: &[u8]) -> u64 {
            // FNV-1a over key || aad || plaintext.
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in std::iter::once(&self.key).chain(aad).chain(plaintext) {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        }
    }

    impl Cipher for ToyCipher {
        fn seal(&self, plaintext: &[u8], aad: &[u8]) -> Vec<u8> {
            let tag = self.tag(aad, plaintext);
            let mut out = tag.to_le_bytes().to_vec();
            out.extend(plaintext.iter().map(|&b| b ^ self.key));
            out
        }

        fn open(&self, ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, CipherError> {
            if ciphertext.len() < 8 {
                return Err(CipherError("truncated".into()));
            }
            let (tag_bytes, body) = ciphertext.split_at(8);
            let stored = u64::from_le_bytes(tag_bytes.try_into().unwrap());
            let plaintext: Vec<u8> = body.iter().map(|&b| b ^ self.key).collect();
            if self.tag(aad, &plaintext) != stored {
                return Err(CipherError("authentication failed".into()));
            }
            Ok(plaintext)
        }
    }

    const SID: StoreId = [7u8; 16];

    #[test]
    fn round_trips_and_returns_authenticated_role_and_generation() {
        let c = ToyCipher { key: 0x5a };
        let sealed = seal_artifact(&c, Role::Snapshot, &SID, 42, b"meeting content");
        let (role, gen, pt) = open_artifact(&c, &SID, &sealed).expect("open");
        assert_eq!(role, Role::Snapshot);
        assert_eq!(gen, 42);
        assert_eq!(pt, b"meeting content");
    }

    #[test]
    fn header_carries_magic_and_version_in_the_clear() {
        let c = ToyCipher { key: 1 };
        let sealed = seal_artifact(&c, Role::Wal, &SID, 0, b"x");
        assert_eq!(&sealed[0..4], &MAGIC);
        assert_eq!(sealed[4], ENC_FORMAT_VERSION);
    }

    #[test]
    fn wrong_key_fails_closed() {
        let sealed = seal_artifact(&ToyCipher { key: 1 }, Role::Checkpoint, &SID, 3, b"secret");
        let err = open_artifact(&ToyCipher { key: 2 }, &SID, &sealed);
        assert!(err.is_err(), "a different key must not open the artifact");
    }

    #[test]
    fn foreign_store_is_rejected() {
        let c = ToyCipher { key: 9 };
        let sealed = seal_artifact(&c, Role::Snapshot, &SID, 1, b"data");
        let other: StoreId = [8u8; 16];
        assert!(open_artifact(&c, &other, &sealed).is_err());
    }

    #[test]
    fn tampered_header_fails_the_open() {
        let c = ToyCipher { key: 9 };
        let mut sealed = seal_artifact(&c, Role::Snapshot, &SID, 1, b"data");
        // Flip the role byte (snapshot -> checkpoint). It is AAD, so the tag no longer matches.
        sealed[5] = Role::Checkpoint as u8;
        assert!(open_artifact(&c, &SID, &sealed).is_err());
    }

    #[test]
    fn tampered_generation_fails_the_open() {
        let c = ToyCipher { key: 9 };
        let mut sealed = seal_artifact(&c, Role::Checkpoint, &SID, 5, b"data");
        sealed[22] ^= 0xff; // mutate the generation field
        assert!(open_artifact(&c, &SID, &sealed).is_err());
    }

    #[test]
    fn tampered_ciphertext_fails_the_open() {
        let c = ToyCipher { key: 9 };
        let mut sealed = seal_artifact(&c, Role::Snapshot, &SID, 1, b"data");
        let last = sealed.len() - 1;
        sealed[last] ^= 0x01;
        assert!(open_artifact(&c, &SID, &sealed).is_err());
    }

    #[test]
    fn unknown_version_is_rejected() {
        let c = ToyCipher { key: 9 };
        let mut sealed = seal_artifact(&c, Role::Snapshot, &SID, 1, b"data");
        sealed[4] = ENC_FORMAT_VERSION + 1;
        let err = open_artifact(&c, &SID, &sealed).unwrap_err();
        assert!(err.0.contains("framing version"), "got: {}", err.0);
    }

    #[test]
    fn too_short_and_bad_magic_are_rejected() {
        let c = ToyCipher { key: 9 };
        assert!(open_artifact(&c, &SID, b"tiny").is_err());
        let mut sealed = seal_artifact(&c, Role::Snapshot, &SID, 1, b"data");
        sealed[0] = b'X';
        assert!(open_artifact(&c, &SID, &sealed).is_err());
    }
}
