from pathlib import Path

import pytest

from src.utils.dataset_pipeline import split_for_image_id


SPLITS = {"train": (1, 700), "validation": (701, 800), "test": (801, 900)}


@pytest.mark.parametrize(
    ("image_id", "expected"), [(1, "train"), (700, "train"), (701, "validation"), (800, "validation"), (801, "test"), (900, "test")],
)
def test_split_boundaries(image_id: int, expected: str) -> None:
    assert split_for_image_id(image_id, SPLITS) == expected


def test_split_rejects_id_outside_public_resplit() -> None:
    with pytest.raises(ValueError, match="no configured split"):
        split_for_image_id(901, SPLITS)
