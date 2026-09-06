"""Registry-ID payload codec for blind extraction.

**Why this exists.** The blind CNN checkpoint (``models/phase8_cnn/phase8_cnn_best.pt``)
has a fixed 64-bit output, was trained only at alpha=0.02, and measures a mean
bit error rate of ~22.2% at that operating point (``docs/phase16_ecc.md``,
``docs/blind_extraction_diagnosis.md``). Embedding arbitrary text directly
needs 48+ bits for even a short word (1-byte length header + UTF-8 body), which
leaves too little of the 64-bit budget for the redundancy that error rate
demands: reaching 95% message-level exact-match via repetition needs ~25x the
raw bit count for a 48-bit message (1200 bits) - about 19x more than the
64-bit budget provides. No error-correcting scheme closes that gap for
arbitrary text on this checkpoint (see ``docs/phase18_id_registry.md``).

**The fix.** Do not embed text. Embed a small fixed-width ID (4 bits, 16
possible values) that indexes into a local message registry; the registry
lookup - not the CNN - is what turns the ID back into text. A 4-bit ID
leaves room for repetition factor r=15 within the 64-bit budget (60 bits
used, 4 spare), which the exact binomial math puts at 96.7% ID-exact-match
probability under an i.i.d.-per-bit-error model at the measured 22.2% BER -
comfortably above the 95% target, with a safety margin against the CNN's
convolutional receptive field (kernel=7) correlating errors between nearby
output positions, which an i.i.d. model can't capture. K=5 (32 entries) was
rejected: it only reaches 90.6%, with no margin left for that same risk.

**Interleaved layout.** The 15 copies of each ID bit are *not* placed
contiguously. They are round-robin interleaved: encoded position
``r * ID_BITS + i`` holds copy ``r`` of ID-bit ``i``, so the 60-bit stream
is 15 repeating groups of ``[bit0, bit1, bit2, bit3]`` rather than 4 blocks
of 15. A first real measurement with the contiguous layout (60/200 exact,
30.5%, ``results/phase18_id_registry``) came in far below the 96.7%
i.i.d. projection - exactly the correlated-error risk the design note above
warned about: the CNN's kernel=7 receptive field smears errors across
*neighbouring* output positions, and a contiguous layout puts all 15 copies
of one ID bit in one neighbourhood, so a single correlated burst can flip
that bit's whole vote. Interleaving spreads each ID bit's 15 copies across
the full 60-bit span so a spatially-local burst lands on multiple different
ID bits' votes (each losing at most a few of 15 copies) instead of wiping
out one bit's vote entirely.

**Degenerate IDs excluded.** ``DEGENERATE_IDS`` (0 = ``0000`` and
``MAX_ENTRIES - 1`` = ``1111``) are never handed out by
``MessageRegistry.register`` - their per-copy value is constant across
*all* 60 encoded bits (not just one bit's 15 copies), so they carry no
internal contrast for the CNN to key off of and, per the same first
measurement, ID 0 (``"hello"``) was the worst performer of the five test
messages (15% exact-match vs. 22.5-42.5% for the others). This is a
registry-allocation policy, not a codec restriction: ``encode_id_bits`` /
``decode_id_bits`` still round-trip every value in
``range(MAX_ENTRIES)`` structurally.

**Balanced XOR mask.** After interleaving fixed the contiguous-block
correlation problem (30.5% -> 77.0% real exact-match, still 5 messages),
a follow-up sweep over **all 16 IDs** (``experiments`` diagnostic, 20 images
each) found a second, distinct mechanism dominating the remaining gap: exact
-match rate is a sharp function of the ID's Hamming weight (count of 1-bits
among its 4 bits), *not* which specific ID it is - weight 2 (e.g. ``0011``,
``0101``, ``0110``) scores 90-100%, weight 1 or 3 scores only 35-70%, and
weight 0/1111 collapse to 0-5%. Because every ID-bit's 15 copies share one
value, an ID's Hamming weight fixes the *global* fraction of 1s in the
64-bit payload at exactly ``weight / 4`` (0%, 23%, 47%, 70%, 94%) -
independent of ``REPETITION``, so more copies cannot fix this. Only
weight 2 (47%) sits near the ~50/50 balance the CNN almost certainly
calibrated on during training (random bit payloads), and interleaving
(a position permutation) cannot change a global 1-count either.

The fix is a **fixed, seeded, globally-balanced XOR mask** (``_ENCODE_MASK``,
30 ones / 30 zeros, built once from ``MASK_SEED`` via
``numpy.random.default_rng`` - never regenerated at runtime) applied to the
60 encoded positions at both encode and decode time. XOR-ing with a known
constant mask is exactly invertible per bit (masking, then a channel bit
flip, then unmasking, is equivalent to the same flip on the unmasked value -
so majority-vote decoding is unaffected in kind), but it decouples the
*physically embedded* stream's 1-fraction from the ID's own Hamming weight:
whatever weight the ID has, roughly half of its mask-XORed positions get
flipped, so the transmitted stream sits near 50% ones regardless of ID.
``MASK_SEED`` must never change once messages are embedded with it - it is
not a security secret, only a fixed codec parameter shared by encode and
decode.

This module holds only the codec (encode/decode) and a small registry
(message <-> ID). It does not touch embedding, the CNN, or the DWT/SVD
mechanics - callers pass the resulting bit list to the existing frozen
``embed()`` exactly like any other payload.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "ID_BITS",
    "REPETITION",
    "ENCODED_BITS",
    "PAYLOAD_BITS",
    "MAX_ENTRIES",
    "DEGENERATE_IDS",
    "USABLE_ENTRIES",
    "MASK_SEED",
    "RegistryError",
    "MessageRegistry",
    "encode_id_bits",
    "decode_id_bits",
    "decode_id_bits_soft",
]

ID_BITS = 4
REPETITION = 15
ENCODED_BITS = ID_BITS * REPETITION  # 60
PAYLOAD_BITS = 64  # the blind CNN's fixed output width; encode_id_bits pads up to this
MAX_ENTRIES = 2**ID_BITS  # 16
DEGENERATE_IDS = frozenset({0, MAX_ENTRIES - 1})  # 0000, 1111 - constant across all 60 bits
USABLE_ENTRIES = MAX_ENTRIES - len(DEGENERATE_IDS)  # 14 - what MessageRegistry can actually allocate

# Fixed codec parameter, NOT a per-run random value and NOT a security secret -
# must stay constant forever so encode and decode always agree on the mask.
# Changing it silently breaks decoding of anything embedded under the old seed.
MASK_SEED = 1729


def _build_balanced_mask(n_bits: int, seed: int) -> np.ndarray:
    """A fixed pseudorandom permutation of ``n_bits // 2`` ones and the rest
    zeros - globally balanced by construction, deterministic given ``seed``."""
    half = n_bits // 2
    mask = np.array([1] * half + [0] * (n_bits - half), dtype=int)
    np.random.default_rng(seed).shuffle(mask)
    return mask


_ENCODE_MASK = _build_balanced_mask(ENCODED_BITS, MASK_SEED)  # 60 bits, 30 ones / 30 zeros


class RegistryError(ValueError):
    """User-correctable problem with a registry lookup or registration."""


class MessageRegistry:
    """A small, fixed-capacity bidirectional map between message text and a
    ``0 <= id < MAX_ENTRIES`` integer. Registration is first-come-first-served
    and stable: once a message is registered its ID never changes."""

    def __init__(self, messages: Sequence[str] | None = None) -> None:
        self._id_to_message: dict[int, str] = {}
        self._message_to_id: dict[str, int] = {}
        for message in messages or ():
            self.register(message)

    def register(self, message: str) -> int:
        """Return the message's ID, registering it in the next free non-degenerate
        slot if new. Never hands out an ID in ``DEGENERATE_IDS``."""
        if message in self._message_to_id:
            return self._message_to_id[message]
        if len(self._id_to_message) >= USABLE_ENTRIES:
            raise RegistryError(
                f"message registry is full ({USABLE_ENTRIES} usable entries); cannot "
                f"register {message!r}. Remove an existing entry or use a shorter "
                f"payload design."
            )
        new_id = 0
        while new_id in self._id_to_message or new_id in DEGENERATE_IDS:
            new_id += 1
        self._id_to_message[new_id] = message
        self._message_to_id[message] = new_id
        return new_id

    def id_for(self, message: str) -> int | None:
        """The message's ID if already registered, else ``None`` (never registers)."""
        return self._message_to_id.get(message)

    def message_for(self, id_: int) -> str | None:
        """The message for ``id_``, or ``None`` if that slot is unregistered."""
        return self._id_to_message.get(id_)

    def __len__(self) -> int:
        return len(self._id_to_message)

    def __contains__(self, message: str) -> bool:
        return message in self._message_to_id


# ---------------------------------------------------------------------------
# Codec: id <-> bits, majority-vote repetition
# ---------------------------------------------------------------------------

def encode_id_bits(id_: int) -> list[int]:
    """Encode ``id_`` (0..15) as ``PAYLOAD_BITS`` (64) bits: the 4 ID bits
    repeated ``REPETITION`` (15) times each and round-robin interleaved -
    encoded position ``r * ID_BITS + i`` holds copy ``r`` of ID-bit ``i``, so
    the 60-bit stream is 15 repeating ``[bit0, bit1, bit2, bit3]`` groups -
    then XORed with the fixed balanced ``_ENCODE_MASK`` (decouples the
    embedded stream's 1-fraction from the ID's own Hamming weight; see the
    module docstring) and zero-padded to 64."""
    if not (0 <= id_ < MAX_ENTRIES):
        raise ValueError(f"id must be in [0, {MAX_ENTRIES}); got {id_}")
    id_bits = [(id_ >> (ID_BITS - 1 - i)) & 1 for i in range(ID_BITS)]
    encoded: list[int] = []
    for _ in range(REPETITION):
        encoded.extend(id_bits)
    masked = [b ^ int(m) for b, m in zip(encoded, _ENCODE_MASK)]
    return masked + [0] * (PAYLOAD_BITS - len(masked))


def decode_id_bits(bits: Sequence[int]) -> tuple[int, float, float]:
    """Majority-vote decode the leading ``ENCODED_BITS`` (60) of ``bits`` back
    to an ID, inverting the mask and interleaved layout from
    ``encode_id_bits``. Returns ``(id, mean_confidence, min_confidence)``
    where each confidence is the fraction of the 15 copies agreeing with the
    winning bit for that ID-bit position, averaged (``mean``) or worst-case
    (``min``) across the 4 ID-bit positions - both in ``[1/15, 1.0]``, higher
    = more decisive vote. Trailing padding bits past ``ENCODED_BITS`` are
    ignored.
    """
    flat = list(bits)[:ENCODED_BITS]
    if len(flat) < ENCODED_BITS:
        raise ValueError(f"need >= {ENCODED_BITS} bits to decode an ID; got {len(flat)}")
    unmasked = [b ^ int(m) for b, m in zip(flat, _ENCODE_MASK)]
    grid = np.asarray(unmasked, dtype=int).reshape(REPETITION, ID_BITS)
    means = grid.mean(axis=0)  # fraction of 1s per ID-bit position, in [0, 1]
    voted = (means >= 0.5).astype(int)
    id_ = 0
    for bit in voted:
        id_ = (id_ << 1) | int(bit)
    # agreement with the winning bit: means if voted 1, else (1 - means)
    agreement = np.where(voted == 1, means, 1.0 - means)
    return id_, float(agreement.mean()), float(agreement.min())


def decode_id_bits_soft(probabilities: Sequence[float]) -> tuple[int, float, float]:
    """Soft-decision counterpart of :func:`decode_id_bits`.

    Takes the decoder's per-bit **probabilities** that the embedded bit is 1
    (not thresholded bits) and combines the 15 copies of each ID bit by summing
    log-likelihood ratios instead of counting hard votes. A copy the decoder is
    unsure about (p near 0.5) contributes almost nothing, while a confident copy
    dominates - whereas :func:`decode_id_bits` gives a 0.51 copy and a 0.99 copy
    exactly one vote each, discarding the margin the decoder actually produced.

    Returns ``(id, mean_posterior, min_posterior)``. Both confidences are
    genuine posterior probabilities that the voted ID bit is correct **under the
    decoder's own per-bit probabilities** (independent-copy assumption), so they
    live in ``[0.5, 1.0]`` and are not comparable to ``decode_id_bits``'s
    vote-agreement fractions - they need their own threshold.

    Measured on the windowed decoder over 840 held-out DIV2K trials: exact ID
    recovery 0.931 (hard) -> 0.973 (soft), and the mean-posterior gate covers
    substantially more decodes at higher precision than the hard-vote gate.
    """
    p = np.asarray(list(probabilities)[:ENCODED_BITS], dtype=float)
    if p.shape[0] < ENCODED_BITS:
        raise ValueError(
            f"need >= {ENCODED_BITS} probabilities to decode an ID; got {p.shape[0]}"
        )
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    # Undo the XOR mask in probability space: where the mask bit is 1 the
    # embedded bit is the ID bit inverted, so the evidence flips with it.
    p_id = np.where(_ENCODE_MASK == 1, 1.0 - p, p)
    llr = np.log(p_id / (1.0 - p_id)).reshape(REPETITION, ID_BITS).sum(axis=0)
    voted = (llr > 0).astype(int)
    id_ = 0
    for bit in voted:
        id_ = (id_ << 1) | int(bit)
    posterior = 1.0 / (1.0 + np.exp(-np.abs(llr)))
    return id_, float(posterior.mean()), float(posterior.min())
