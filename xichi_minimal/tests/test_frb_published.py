"""The published nine-burst limit is reproduced by the published method."""
import pytest

import frb_published as fp


def test_table_is_the_published_one():
    assert len(fp.WEI_WU_TABLE) == 9
    assert fp.WEI_WU_TABLE[0][:3] == ("FRB121102", 0.19273, 558.1)


def test_reproduces_the_published_limits():
    """Priors are not stated in the paper, so agreement to 25% is the test."""
    out = fp.gaussian_posterior(fp.WEI_WU_TABLE)
    assert out["limit68"] == pytest.approx(fp.PUBLISHED_68, rel=0.25)
    assert out["limit95"] == pytest.approx(fp.PUBLISHED_95, rel=0.25)


def test_more_bursts_do_not_tighten_this_method():
    """With a fitted intrinsic scatter the limit saturates: 98 bursts give
    about the same number as nine."""
    nine = fp.gaussian_posterior(fp.WEI_WU_TABLE)["limit95"]
    many = fp.gaussian_posterior(fp.table_from_repository())["limit95"]
    assert many == pytest.approx(nine, rel=0.3)


def test_same_nine_bursts_are_in_the_repository_table():
    rows = fp.table_from_repository([b[0] for b in fp.WEI_WU_TABLE])
    assert len(rows) == 9
