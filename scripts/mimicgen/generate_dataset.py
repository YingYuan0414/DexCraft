"""Step 6 of the MimicGen pipeline: generate demos with MimicGen.

Thin wrapper around mimicgen/scripts/generate_dataset.py that registers the DexCraft envs, configs and
env interface first. All arguments are forwarded, e.g.

    python scripts/mimicgen/generate_dataset.py --config configs/mimicgen/spray_bottle.json --auto-remove-exp
"""
import runpy

import dexcraft.mimicgen  # noqa: F401

if __name__ == "__main__":
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__", alter_sys=True)
