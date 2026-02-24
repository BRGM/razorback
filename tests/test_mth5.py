import pathlib

from razorback import Inventory
from razorback.io import load_mth5, clean_whitespace


HERE = pathlib.Path(__file__).parent.resolve()


def test_synthetic():
    feed = load_mth5(
        HERE / "data/synthetic_test_data.h5",
        "{survey}_{station}_{channel}",
        clean_field = clean_whitespace("_"),
    )
    inv = Inventory(feed)
    assert inv.tags == {
        'EMTF_Synthetic_test1_ex',
        'EMTF_Synthetic_test1_ey',
        'EMTF_Synthetic_test1_hx',
        'EMTF_Synthetic_test1_hy',
        'EMTF_Synthetic_test1_hz',
        'EMTF_Synthetic_test2_ex',
        'EMTF_Synthetic_test2_ey',
        'EMTF_Synthetic_test2_hx',
        'EMTF_Synthetic_test2_hy',
        'EMTF_Synthetic_test2_hz',
    }


def test_small_template():
    feed = load_mth5(
        HERE / "data/synthetic_test_data.h5",
        "{station}_{channel}",
        clean_field = clean_whitespace("_"),
    )
    inv = Inventory(feed)
    assert inv.tags == {
        'test1_ex',
        'test1_ey',
        'test1_hx',
        'test1_hy',
        'test1_hz',
        'test2_ex',
        'test2_ey',
        'test2_hx',
        'test2_hy',
        'test2_hz',
    }
