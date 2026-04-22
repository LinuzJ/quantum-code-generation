import argparse
import random
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a reproducible subset of instance indices.")
    p.add_argument("--num-instances", type=int, default=580, help="Total number of instances available")
    p.add_argument("--k", type=int, required=True, help="Number of indices to select")
    p.add_argument(
        "--mode",
        choices=["random", "first"],
        default="random",
        help="Subset selection mode",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random mode")
    p.add_argument("--out", required=True, help="Output path (one index per line)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = int(args.num_instances)
    k = int(args.k)
    if k <= 0:
        raise SystemExit("k must be > 0")
    if k > n:
        raise SystemExit(f"k={k} is larger than num-instances={n}")

    if args.mode == "first":
        indices: List[int] = list(range(k))
    else:
        rng = random.Random(int(args.seed))
        indices = rng.sample(list(range(n)), k)
        indices.sort()

    with open(args.out, "w", encoding="utf-8") as f:
        for i in indices:
            f.write(f"{i}\n")

    print(f"Wrote {len(indices)} indices to {args.out}")


if __name__ == "__main__":
    main()
