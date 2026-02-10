"""Auto-generated per-instance evaluator.

Compares:
  (i) reference ground-truth circuit (dataset) and
  (ii) quasar-generated circuit

After compiling both onto the same IBM Fake backend using the same transpilation settings.
"""

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
        index=0,
        fake_backend="FakeKyoto",
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
