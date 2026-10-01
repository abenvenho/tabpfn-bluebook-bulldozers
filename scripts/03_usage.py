"""Read the Prior Labs API usage for this token — free, spends no quota.

Usage: TABPFN_TOKEN=... python scripts/03_usage.py [--tag LABEL]

Prints current usage / limit / reset time and appends a timestamped line to
results/usage_log.txt, so the real cost of each block is the difference
between two readings (before and after). Safe to run while another block is
running in a separate terminal.
"""
import argparse
import datetime
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="label for this reading")
    args = ap.parse_args()

    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from tabpfn_client.client import ServiceClient
        raw = ServiceClient.get_api_usage(tabpfn_client.get_access_token())
        line = (f"{now}\t{args.tag}\tused={raw.get('current_usage')}\t"
                f"limit={raw.get('usage_limit')}\treset={raw.get('reset_time')}")
    except Exception as e_raw:  # fall back to the public, formatted helper
        try:
            line = f"{now}\t{args.tag}\t{tabpfn_client.get_api_usage()}"
        except Exception as e_pub:
            line = f"{now}\t{args.tag}\tERROR raw={e_raw!r} public={e_pub!r}"

    print(line)
    Path("results").mkdir(exist_ok=True)
    with open("results/usage_log.txt", "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
