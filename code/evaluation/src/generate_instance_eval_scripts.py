import argparse
import os
from textwrap import dedent


TEMPLATE = """\
\"\"\"Auto-generated per-instance evaluator.

Compares:
  (i) reference ground-truth circuit (dataset) and
  (ii) quasar-generated circuit

After compiling both onto the same IBM Fake backend using the same transpilation settings.
\"\"\"

import os
import sys


def _add_src_to_path():
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, os.pardir))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


ROOT_DIR = _add_src_to_path()

from instance_eval_common import EvalConfig, evaluate_instance  # noqa: E402


def main():
    cfg = EvalConfig(
        quasar_json=os.path.join(ROOT_DIR, "generated_circuits", "quasar.json"),
        index={index},
        fake_backend="FakeTorino",
        seed_transpiler=0,
        optimization_level=3,
        layout_method="sabre",
        routing_method="sabre",
        translation_method=None,
        scheduling_method=None,
        approximation_degree=1.0,
        out_dir=os.path.join(ROOT_DIR, "out_instance_eval"),
    )
    evaluate_instance(cfg)


if __name__ == "__main__":
    main()
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate per-instance evaluation scripts")
    p.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="How many instance scripts to generate (e.g. 580)",
    )
    p.add_argument(
        "--out-dir",
        default="instance_eval_scripts",
        help="Output directory (relative to repo root)",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_dir = os.path.join(root_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for index in range(args.num_instances):
        out_path = os.path.join(out_dir, f"instance_{index:04d}.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(dedent(TEMPLATE.format(index=index)))

    print(f"Generated {args.num_instances} per-instance scripts in: {out_dir}")


if __name__ == "__main__":
    main()
