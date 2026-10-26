"""Refuse a Slurm job that could still be running when a /scratch purge window starts.

    python3 slurm/purge_check.py --now EPOCH --limit HH:MM:SS [--file PATH]

The latest end is now + limit. Exit 0: clear of every window; stdout then holds the start of the
next window in local time (YYYY-MM-DDTHH:MM:SS), the --deadline for sbatch, or nothing when none
is left.
Exit 1: refused. Exit 2: a bad limit or dates file. Runs on Explorer's system python3 (3.9), so
it uses the standard library only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

DEFAULT_FILE = "slurm/purge_dates"


def limit_seconds(text: str) -> int:
    """Seconds in a Slurm time limit: D-HH:MM:SS, D-HH:MM, HH:MM:SS or MM:SS."""
    rest, days = text.strip(), 0
    if "-" in rest:
        day_text, rest = rest.split("-", 1)
        if not day_text.isdigit():
            raise ValueError(f"bad time limit {text!r}")
        days = int(day_text)
    parts = rest.split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"bad time limit {text!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:  # MM:SS, or HH:MM after a day count
        numbers = numbers + [0] if "-" in text else [0, *numbers]
    hours, minutes, seconds = numbers
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def read_windows(path: str) -> list[tuple[datetime, datetime]]:
    """(start, end) pairs from a dates file: `START END` per line, ISO 8601 with a UTC offset."""
    windows = []
    with open(path, encoding="utf-8") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected START END")
            start, end = (datetime.fromisoformat(field) for field in fields)
            if start.tzinfo is None or end.tzinfo is None:
                raise ValueError(f"{path}:{number}: each time needs a UTC offset")
            if end <= start:
                raise ValueError(f"{path}:{number}: the window ends before it starts")
            windows.append((start, end))
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--now", type=int, required=True, help="seconds since the epoch")
    parser.add_argument("--limit", required=True, help="this job's time limit")
    parser.add_argument("--file", default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    try:
        seconds = limit_seconds(args.limit)
        windows = read_windows(args.file)
    except (OSError, ValueError) as error:
        print(f"purge_check: {error}", file=sys.stderr)
        return 2
    # Explorer's python3 is 3.9, which has no datetime.UTC
    now = datetime.fromtimestamp(args.now, tz=timezone.utc)  # noqa: UP017
    latest = now + timedelta(seconds=seconds)
    for start, end in windows:
        if now < end and latest > start:
            print(
                f"refused: the job could run until {latest.astimezone(start.tzinfo).isoformat()}"
                f" and a /scratch purge window starts {start.isoformat()}",
                file=sys.stderr,
            )
            return 1
    upcoming = [start for start, _ in windows if start > now]
    if upcoming:
        print(min(upcoming).astimezone().strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
