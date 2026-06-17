#!/usr/bin/env bash
#
# reproduce.sh — regenerate every figure and the Table 3 numbers of the paper
# from the bundled fit products.  No external archive access is required: the
# Koester (2010) DA grid is shipped in compact resampled form as
# data/koester_cache.npz, so the white-dwarf component is rebuilt without the
# full 140 MB model grid.
#
# Usage:
#   pip install -r requirements.txt
#   ./reproduce.sh                 # uses `python3`
#   PYTHON=/path/to/python ./reproduce.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export CYC_ROOT="$HERE"
PY="${PYTHON:-python3}"

echo "== Table 3 numbers =="
( cd "$HERE/code" && "$PY" reproduce_numbers.py )

echo; echo "== main-pipeline figures =="
cd "$HERE/code"
"$PY" plot_profile_curves.py        # figures/bench_validation.pdf   (Fig 1)
"$PY" lightcurve_panels.py          # figures/lightcurve_panels.pdf  (Fig 2)
"$PY" emission_lines.py             # figures/emission_lines.pdf
"$PY" make_corner.py                # figures/J0005_corner_cyc.pdf   (appendix)

echo; echo "== joint-fit / branch figures =="
cd "$HERE/validation_candidates/full_refit/code"
"$PY" regen_joint_fullrefit.py      # figures/{src}_joint_fit.pdf    (Fig 3)
"$PY" plot_full_refit_profiles.py   # figures/B_profiles_full_refit.pdf (Fig 4)
                                    # + validation_sdss_full_refit.pdf
"$PY" plot_branch_decomposition.py  # branch_decomposition_combined + J0035 (App. C)

# gather the full_refit figures next to the others for convenience
cp "$HERE"/validation_candidates/full_refit/figures/*.pdf "$HERE/figures/" 2>/dev/null || true

echo; echo "Done.  All figures are in $HERE/figures/ ."
