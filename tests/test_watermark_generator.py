import uuid

import pytest

from src.watermark.watermark_generator import (
    SUPPORTED_BIT_LENGTHS,
    bits_to_bytes,
    bytes_to_bits,
    generate_from_text,
    generate_from_uuid,
    generate_random,
    text_to_bits,
    validate_bit_length,
)


FIXED_UUID = "12345678-1234-5678-1234-567812345678"


@pytest.mark.parametrize("bit_length", sorted(SUPPORTED_BIT_LENGTHS))
def test_uuid_payload_has_requested_binary_length(bit_length: int) -> None:
    payload = generate_from_uuid(FIXED_UUID, bit_length)
    assert len(payload) == bit_length
    assert set(payload) <= {0, 1}


def test_uuid_generation_is_deterministic() -> None:
    assert generate_from_uuid(FIXED_UUID, 256) == generate_from_uuid(FIXED_UUID, 256)
    assert generate_from_uuid(FIXED_UUID, 256) != generate_from_uuid(uuid.uuid4(), 256)


def test_random_generation_is_reproducible_from_returned_identifier() -> None:
    identifier, payload = generate_random(128)
    assert payload == generate_from_uuid(identifier, 128)


@pytest.mark.parametrize("length", [0, 7, 9, 300, -8])
def test_invalid_length_is_rejected(length: int) -> None:
    with pytest.raises(ValueError, match="Unsupported watermark length"):
        validate_bit_length(length)


def test_invalid_uuid_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_from_uuid("not-a-uuid", 64)


def test_utf8_text_conversion_and_round_trip() -> None:
    bits = text_to_bits("TEAM-水")
    assert bits_to_bytes(bits).decode("utf-8") == "TEAM-水"
    assert len(generate_from_text("TEAM-水", 512)) == 512


@pytest.mark.parametrize("value", ["", None])
def test_empty_or_non_string_text_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        text_to_bits(value)  # type: ignore[arg-type]


def test_bit_conversion_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="only 0 or 1"):
        bits_to_bytes([0, 1, 2])
