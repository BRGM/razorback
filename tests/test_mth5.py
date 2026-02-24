from razorback import Inventory
from razorback.io import load_mth5, clean_whitespace


def test_synthetic():
    feed = load_mth5(
        "data/synthetic_test_data.h5",
        "{survey}_{station}_{channel}",
        clean_field = clean_whitespace("_"),
    )
    inv = Inventory(feed)
    assert inv.tags == []
