"""Build the committed BTC slice that ``envs/btc_market.py`` reads.

The source is minute-resolution OHLCV, which is far too large to check in — the
full history this was cut from is 357 MB. This resamples it to hourly bars and
writes a gzipped CSV of a few tens of thousands of rows, small enough to commit
and long enough to contain more than one market regime.

    python examples/data/make_btc_slice.py \\
        --source '/home/erels/PycharmProjects/BtcSwarm/envs/btc-data-1[5-9].csv' \\
        --source '/home/erels/PycharmProjects/BtcSwarm/envs/btc-data-20.csv'

The default sources are the paths this slice was originally cut from, which exist
only on the machine it was built on; the committed output is what the environment
actually uses, so this script is here for provenance and for cutting a different
window, not as a build step.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Sequence

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "btc_1h.csv.gz")
#: Where the committed slice was cut from. Machine-specific by nature.
DEFAULT_SOURCES = (
    "/home/erels/PycharmProjects/BtcSwarm/envs/btc-data-1[5-9].csv",
    "/home/erels/PycharmProjects/BtcSwarm/envs/btc-data-20.csv",
)
COLUMNS = ("Timestamp", "Open", "High", "Low", "Close", "Volume")


def resolve(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise SystemExit(f"no files matched {pattern!r}")
        paths += matched
    return sorted(dict.fromkeys(paths))


def build(paths: Sequence[str], freq: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, usecols=list(COLUMNS)) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["Timestamp"], unit="s", utc=True)
    raw = raw.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")

    bars = raw.resample(freq).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"})
    bars = bars.dropna(subset=["Open", "Close"])
    bars.columns = [name.lower() for name in bars.columns]
    return bars.round(2).reset_index()


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", action="append",
                        help="minute-CSV path or glob (repeatable)")
    parser.add_argument("--freq", default="1h", help="bar size (default 1h)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve(args.source or DEFAULT_SOURCES)
    bars = build(paths, args.freq)
    bars.to_csv(args.out, index=False, compression="gzip")
    size = os.path.getsize(args.out) / 1e6
    span = f"{bars['timestamp'].iloc[0].date()} -> {bars['timestamp'].iloc[-1].date()}"
    print(f"{len(bars):,} {args.freq} bars, {span}, "
          f"close {bars['close'].min():,.0f}-{bars['close'].max():,.0f}, "
          f"{size:.2f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
