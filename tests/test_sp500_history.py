import pandas as pd

from equity_agent.data.sp500_history import ever_members, members_asof, membership_mask

# Synthetic change log (no network):
#   2020-01-01: CCC added, YYY removed  -> before 2020, CCC not a member, YYY was
#   2022-01-01: BBB added, XXX removed  -> before 2022, BBB not a member, XXX was
CURRENT = ["AAA", "BBB", "CCC"]
CHANGES = pd.DataFrame(
    {
        "eff": pd.to_datetime(["2020-01-01", "2022-01-01"]),
        "add": ["CCC", "BBB"],
        "rem": ["YYY", "XXX"],
    }
)


def test_members_asof_today_equals_current() -> None:
    assert members_asof("2026-01-01", CURRENT, CHANGES) == set(CURRENT)


def test_members_asof_undoes_future_changes() -> None:
    # Between the two changes: undo only the 2022 change (remove BBB, add XXX back).
    assert members_asof("2021-06-01", CURRENT, CHANGES) == {"AAA", "CCC", "XXX"}
    # Before both changes: undo both (remove BBB & CCC, add XXX & YYY back).
    assert members_asof("2019-06-01", CURRENT, CHANGES) == {"AAA", "XXX", "YYY"}


def test_ever_members_is_union_with_removed() -> None:
    union = ever_members(CURRENT, CHANGES, "2015-01-01")
    # current + everything removed since start
    assert set(union) == {"AAA", "BBB", "CCC", "XXX", "YYY"}


def test_membership_mask_tracks_segments() -> None:
    idx = pd.Index(pd.to_datetime(pd.date_range("2018-06-30", "2024-06-30", freq="YE")).date)
    mask = membership_mask(idx, CURRENT, CHANGES)
    # columns = union of ever-members
    assert set(mask.columns) == {"AAA", "BBB", "CCC", "XXX", "YYY"}
    by_date = {pd.Timestamp(d): mask.loc[d] for d in mask.index}

    def members(ts: pd.Timestamp) -> set[str]:
        row = by_date[ts]
        return set(row[row].index)

    # 2018-12-31: before both changes -> {AAA, XXX, YYY}
    assert members(pd.Timestamp("2018-12-31")) == {"AAA", "XXX", "YYY"}
    # 2020-12-31: after 2020 change, before 2022 -> {AAA, CCC, XXX}
    assert members(pd.Timestamp("2020-12-31")) == {"AAA", "CCC", "XXX"}
    # 2022-12-31 onward: current set
    assert members(pd.Timestamp("2022-12-31")) == {"AAA", "BBB", "CCC"}
