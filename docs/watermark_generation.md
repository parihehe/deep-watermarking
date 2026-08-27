# Watermark Generation — Phase 3

## Purpose

The baseline embeds a binary payload, not a manually drawn watermark image. Each experiment can therefore be repeated from a recorded UUID or text input.

## UUID-derived payloads

1. Generate or provide a canonical UUID.
2. Encode its canonical string as ASCII.
3. Compute SHA-256 over `counter || seed`, starting at counter zero.
4. Concatenate digest blocks until the requested size is met.
5. Convert bytes to most-significant-bit-first values and truncate exactly to the requested length.

The counter expansion is needed because SHA-256 is 256 bits while the project supports 512- and 1024-bit payloads. It is deterministic: the same UUID and bit length always produce exactly the same bits.

## Text payloads

Text is encoded as UTF-8. `text_to_bits()` returns its direct raw representation, while `generate_from_text()` hashes/expands it to a configured fixed length. Direct text bits are useful for later human-readable experiments; fixed-length text-derived bits make baseline capacity sweeps comparable.

## Supported raw lengths

`8, 16, 32, 64, 128, 256, 512, 1024` bits.

Phase 3 does **not** apply ECC, encryption, scrambling, or embedding. ECC will be an explicitly switchable and separately evaluated Phase 15 module, so raw BER and capacity remain scientifically interpretable.
