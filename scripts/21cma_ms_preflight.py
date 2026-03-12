#!/usr/bin/env python3

import argparse
from collections import Counter

from casacore.tables import table


def unique_times(values):
    out = []
    for value in values:
        if not out or value != out[-1]:
            out.append(value)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the metadata shape of a 21CMA measurement set."
    )
    parser.add_argument("ms", help="Path to the measurement set")
    parser.add_argument(
        "--time-sample-rows",
        type=int,
        default=820 * 300,
        help="Rows to sample from the main table for a quick time-axis summary",
    )
    args = parser.parse_args()

    with table(args.ms, readonly=True) as ms:
        nrows = ms.nrows()
        time_sample_rows = min(args.time_sample_rows, nrows)
        times = ms.getcol("TIME", 0, time_sample_rows)
        unique = unique_times(times)

        rows_per_timestep = 0
        if len(times):
            first_time = times[0]
            for value in times:
                if value == first_time:
                    rows_per_timestep += 1
                else:
                    break

        diffs = [round(unique[i + 1] - unique[i], 6) for i in range(len(unique) - 1)]
        diff_counts = Counter(diffs)

    with table(f"{args.ms}/SPECTRAL_WINDOW", readonly=True) as spw:
        chan_freq = spw.getcell("CHAN_FREQ", 0)
        chan_width = spw.getcell("CHAN_WIDTH", 0)

    with table(f"{args.ms}/POLARIZATION", readonly=True) as pol:
        corr_type = pol.getcell("CORR_TYPE", 0)

    print(f"MS: {args.ms}")
    print(f"nrows: {nrows}")
    print(f"rows_per_timestep: {rows_per_timestep}")
    print(f"sampled_unique_times: {len(unique)}")
    if unique:
        print(f"time_first: {unique[0]}")
        print(f"time_last: {unique[-1]}")
    print(f"first_time_diffs: {diffs[:10]}")
    print(f"top_time_diffs: {diff_counts.most_common(10)}")
    print(f"corr_type: {list(corr_type)}")
    print(f"nchan: {len(chan_freq)}")
    print(f"freq_start_hz: {chan_freq[0]}")
    print(f"freq_end_hz: {chan_freq[-1]}")
    print(f"chan_width_hz: {chan_width[0]}")


if __name__ == "__main__":
    main()
