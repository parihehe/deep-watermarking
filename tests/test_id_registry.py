"""Unit tests for the registry-ID payload codec (src/watermark/id_registry.py).

No image, no CNN, no embedding here - just the bit-level codec and the
registry's bookkeeping, matching the feasibility math this design was chosen
from (K=4 ID bits, r=15 repetition, 60/64 bits used).
"""

from __future__ import annotations

import pytest

from src.watermark import id_registry as idreg


def test_encode_length_and_padding():
    bits = idreg.encode_id_bits(5)
    assert len(bits) == idreg.PAYLOAD_BITS == 64
    assert bits[idreg.ENCODED_BITS :] == [0] * (idreg.PAYLOAD_BITS - idreg.ENCODED_BITS)


def test_encode_rejects_out_of_range_id():
    with pytest.raises(ValueError):
        idreg.encode_id_bits(-1)
    with pytest.raises(ValueError):
        idreg.encode_id_bits(idreg.MAX_ENTRIES)


def test_mask_breaks_the_historically_degenerate_all_zero_and_all_one_patterns():
    # Before the fixed XOR mask existed, id 0 (0000) encoded to 60 zero bits
    # and id 15 (1111) to 60 one bits - the exact global-1-fraction imbalance
    # that made them (and every weight != 2 id) the worst performers on the
    # real CNN (see module docstring / results/phase18_id_registry). The mask
    # must destroy that pattern: both must now carry exactly 30 ones (the
    # mask's own weight), not 0 or 60.
    zeros_encoded = idreg.encode_id_bits(0)[: idreg.ENCODED_BITS]
    ones_encoded = idreg.encode_id_bits(idreg.MAX_ENTRIES - 1)[: idreg.ENCODED_BITS]
    assert zeros_encoded != [0] * idreg.ENCODED_BITS
    assert ones_encoded != [1] * idreg.ENCODED_BITS
    assert sum(zeros_encoded) == 30
    assert sum(ones_encoded) == 30


def test_mask_is_fixed_and_reproducible_from_the_documented_seed():
    # Regenerating from MASK_SEED must reproduce the exact same mask bit for
    # bit - guards against someone reseeding at import time (e.g. switching to
    # a time- or process-based seed) and silently breaking every
    # already-embedded image, since encode and decode would then disagree.
    rebuilt = idreg._build_balanced_mask(idreg.ENCODED_BITS, idreg.MASK_SEED)
    assert list(rebuilt) == list(idreg._ENCODE_MASK)
    assert sum(rebuilt) == idreg.ENCODED_BITS // 2  # globally balanced: 30 ones / 30 zeros


@pytest.mark.parametrize("id_", range(idreg.MAX_ENTRIES))
def test_clean_round_trip_every_id(id_: int):
    bits = idreg.encode_id_bits(id_)
    decoded, mean_conf, min_conf = idreg.decode_id_bits(bits)
    assert decoded == id_
    assert mean_conf == pytest.approx(1.0)
    assert min_conf == pytest.approx(1.0)


def test_majority_vote_tolerates_a_minority_of_flipped_copies():
    bits = idreg.encode_id_bits(9)  # 1001
    corrupted = list(bits)
    # flip 7 of the 15 interleaved copies of the first ID bit (still a minority: 7 < 8)
    # copy r of ID-bit 0 lives at position r * ID_BITS (see encode_id_bits)
    for r in range(7):
        corrupted[r * idreg.ID_BITS] ^= 1
    decoded, _, _ = idreg.decode_id_bits(corrupted)
    assert decoded == 9


def test_majority_vote_flips_when_majority_of_copies_corrupted():
    bits = idreg.encode_id_bits(9)  # 1001
    corrupted = list(bits)
    # flip 8 of the 15 interleaved copies of the first ID bit (majority: 8 > 7) -> that bit flips
    for r in range(8):
        corrupted[r * idreg.ID_BITS] ^= 1
    decoded, _, _ = idreg.decode_id_bits(corrupted)
    assert decoded == 1  # 0001, the leading bit flipped from 1 to 0


def test_decode_ignores_trailing_padding():
    bits = idreg.encode_id_bits(3)
    noisy_padding = bits[: idreg.ENCODED_BITS] + [1] * (idreg.PAYLOAD_BITS - idreg.ENCODED_BITS)
    decoded, _, _ = idreg.decode_id_bits(noisy_padding)
    assert decoded == 3


def test_decode_rejects_too_few_bits():
    with pytest.raises(ValueError):
        idreg.decode_id_bits([0] * (idreg.ENCODED_BITS - 1))


def test_registry_registers_and_looks_up():
    # ID 0 is degenerate and skipped; the first two registrations get 1 and 2.
    reg = idreg.MessageRegistry(["hello", "hi"])
    assert reg.id_for("hello") == 1
    assert reg.id_for("hi") == 2
    assert reg.message_for(1) == "hello"
    assert reg.message_for(2) == "hi"
    assert reg.id_for("unknown") is None
    assert reg.message_for(15) is None
    assert len(reg) == 2


def test_registry_never_hands_out_a_degenerate_id():
    reg = idreg.MessageRegistry([f"msg{i}" for i in range(idreg.USABLE_ENTRIES)])
    assert set(reg._id_to_message) & idreg.DEGENERATE_IDS == set()


def test_registry_register_is_idempotent():
    reg = idreg.MessageRegistry()
    first = reg.register("Vestigia")
    second = reg.register("Vestigia")
    assert first == second == 1  # ID 0 is degenerate and skipped
    assert len(reg) == 1


def test_registry_capacity_enforced():
    reg = idreg.MessageRegistry([f"msg{i}" for i in range(idreg.USABLE_ENTRIES)])
    assert len(reg) == idreg.USABLE_ENTRIES
    with pytest.raises(idreg.RegistryError):
        reg.register("one too many")


def test_registry_and_codec_compose_end_to_end():
    reg = idreg.MessageRegistry(["hello", "hi", "owner-2026", "Vestigia", "日本語"])
    for message in ("hello", "hi", "owner-2026", "Vestigia", "日本語"):
        id_ = reg.id_for(message)
        bits = idreg.encode_id_bits(id_)
        decoded_id, conf, _ = idreg.decode_id_bits(bits)
        assert reg.message_for(decoded_id) == message
        assert conf == pytest.approx(1.0)
