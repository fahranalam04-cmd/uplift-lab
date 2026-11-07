"""Command-line entrypoint: ``uplift-lab <stage>`` or ``uplift-lab all``.

Stages (each writes JSON under reports/):
  prepare   download + checksum the Criteo CSV, convert to Parquet   -> data_summary.json
  readout   SRM, balance, ITT, CACE, CUPAC variance reduction         -> readout.json
  simulate  A/A, peeking vs mSPRT, power validation                  -> simulations.json
  uplift    S/T/X/transformed-outcome learners, Qini, uplift@k       -> uplift.json
  figures   PNG figures for README / report                          -> reports/figures/
  report    render reports/readout.md from the JSON files
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uplift-lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "readout", "simulate", "uplift", "figures",
                                          "report", "all"])
    parser.add_argument("--quick", action="store_true",
                        help="fewer simulations / smaller samples (smoke test)")
    args = parser.parse_args(argv)
    stages = (["prepare", "readout", "simulate", "uplift", "figures", "report"]
              if args.stage == "all" else [args.stage])
    for stage in stages:
        t0 = time.time()
        print(f"== {stage}", flush=True)
        if stage == "prepare":
            from . import data
            data.prepare()
        elif stage == "readout":
            from . import readout
            readout.run()
        elif stage == "simulate":
            from . import simulate
            simulate.run(quick=args.quick)
        elif stage == "uplift":
            from . import uplift_study
            uplift_study.run(quick=args.quick)
        elif stage == "figures":
            from . import figures
            figures.run()
        elif stage == "report":
            from . import report
            report.run()
        print(f"== {stage} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
