"""
Generate passkeys for the 3DPharma app.

Usage:
  python generate_passkey.py                        # 1 lifetime key
  python generate_passkey.py --hours 2              # 1 key valid 2h from first use
  python generate_passkey.py --hours 24 --count 10  # 10 keys valid 24h
  python generate_passkey.py --hours 2 --count 500  # 500 keys valid 2h
  python generate_passkey.py --count 500            # 500 lifetime keys
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from auth import generate_passkey

parser = argparse.ArgumentParser(description="Generate 3DPharma passkeys.")
parser.add_argument("--hours", type=float, default=None,
                    help="Validity in hours from first use. Omit for lifetime.")
parser.add_argument("--count", type=int, default=1,
                    help="Number of passkeys to generate (default: 1).")
args = parser.parse_args()

label = f"{args.hours}h from first use" if args.hours else "lifetime"
print(f"\nGenerating {args.count} passkey(s) [{label}]:\n")

for i in range(args.count):
    print(generate_passkey(duration_hours=args.hours))
