//! Arena block layout arithmetic — how many 4-byte words a node occupies.
//!
//! Split out of `hnsw/mod.rs` as pure code motion.

use super::*;

/// Header size, in 4-byte units, of one node block. See [`HNSWIndex::nodes`].
///
/// `count | norm | m0 + 1 neighbour slots`, rounded up to a 16-byte boundary so the
/// vector that follows stays SIMD-aligned. The spare neighbour slot lets insertion
/// push-then-prune in place rather than spilling to a `Vec`.
#[inline(always)]
pub(super) const fn node_hdr_len(m0: usize) -> usize {
    (2 + m0 + 1).div_ceil(4) * 4
}

/// Words needed for `dim` packed RaBitQ sign bits (1 bit/dim, byte-packed then word-rounded).
///
/// Composed as `dim -> bytes -> words` (not a single `div_ceil(32)`) so it matches exactly
/// how [`RaBitCode::bits`](crate::vector::rabitq::RaBitCode::bits) packs — `bits[i/8]` bit
/// `i%8` — which is byte-granular, not word-granular.
#[inline(always)]
pub(super) const fn rabitq_bit_words(dim: usize) -> usize {
    dim.div_ceil(8).div_ceil(4)
}

/// Size, in 4-byte units, of the vector region of a node block.
///
/// `F32` needs one word per dimension. `SQ8` packs one byte per dimension, rounded up to a
/// whole word. `RaBitQ` packs one *bit* per dimension plus two `f32` scalars (`dtc_sq`,
/// `est_factor`) that its distance estimator needs per vector — see [`Storage::RaBitQ`]. In
/// every case the codes sit in the same arena as the links, so a node visit still touches
/// exactly one contiguous block.
/// Words needed for `dim` nibble-packed codes (4 bits/dim, byte-packed then word-rounded) —
/// TurboQuant's MSE indices. Byte-granular like [`rabitq_bit_words`], for the same reason.
#[inline(always)]
pub(super) const fn nibble_words(dim: usize) -> usize {
    dim.div_ceil(2).div_ceil(4)
}

/// `quant_bits` is the active multi-bit budget: `turbo_bits` under `TurboQuant`,
/// `rabit_bits` under `TurboRabit`, ignored (pass 0) otherwise — see
/// [`HNSWConfig::quant_bits`].
#[inline(always)]
#[allow(deprecated)] // internal handling of Storage::TurboQuant until its 0.8 removal
pub(super) const fn vec_words(storage: Storage, dim: usize, quant_bits: usize) -> usize {
    match storage {
        Storage::F32 => dim,
        Storage::SQ8 => dim.div_ceil(4),
        Storage::RaBitQ => 2 + rabitq_bit_words(dim),
        // `[gamma][qjl sign bits][mse nibbles]` — the nibble section exists only when
        // there are MSE bits (`total_bits > 1`); `total_bits = 1` is a pure QJL sketch.
        // Nibbles are sized by the FWHT-PADDED dim (next power of two): the structured
        // rotation quantizes in the padded space, so `TurboCode::idx` is padded-length.
        // The qjl section stays at the raw dim (the sketch runs over the unpadded residual).
        Storage::TurboQuant => {
            1 + rabitq_bit_words(dim)
                + if quant_bits > 1 {
                    nibble_words(crate::vector::turboquant::fht_padded_dim(dim))
                } else {
                    0
                }
        }
        // `[dtc_sq][f_rescale][nibble-packed codes]` — one 4-bit slot per coordinate
        // regardless of B (bounded 1..=4 at fit time), read by the FUSED kernel: the
        // Extended-RaBitQ estimate is linear in the code value, so one cvt+FMA per
        // coordinate replaces the earlier B bit-plane passes. `quant_bits` no longer
        // affects the block size, only the values inside the slots.
        Storage::TurboRabit => 2 + nibble_words(dim),
        // Byte-identical to TurboRabit: Warren's walk IS TurboRabit's walk. Its extra 8-bit
        // residual lives in a cold side array, not the hot block.
        Storage::Warren => 2 + nibble_words(dim),
    }
}

/// Size, in 4-byte units, of one node block.
#[inline(always)]
pub(super) const fn node_stride(
    m0: usize,
    dim: usize,
    storage: Storage,
    quant_bits: usize,
) -> usize {
    node_hdr_len(m0) + vec_words(storage, dim, quant_bits)
}
