#!/usr/bin/env bash
# Maintainer check for WSL2/Linux with an NVIDIA GPU: every registered lab end to end.
#   bash tools/verify_all.sh            # test + bench, reference on CPU and CUDA, student skeletons
#   bash tools/verify_all.sh --quick    # tests only (no benchmarks)
# For each lab in tools/lab_meta.py that exists in this checkout:
#   - reference: --mode test (and bench) on cpu and, where the lab has CUDA parts, on cuda
#   - student skeleton: --part A must build and fail at a TODO (exit 1, never 3)
#   - tools/fault_inject.py --lab N when that tool exists
# then workload_audit, check_links, math_check (if present), format_check, test_platform.
# Writes nothing outside labs/results/. Exit code 1 if anything failed.
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
QUICK=0
[[ ${1:-} == --quick ]] && QUICK=1
PY=${PYTHON:-python}
LOG=results/verify-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
failed=()
passed=0

run() { # run <label> <expected rc> <command...>
    local label=$1 want=$2
    shift 2
    local out="$LOG/${label//[^A-Za-z0-9_.-]/_}.log"
    "$@" > "$out" 2>&1
    local rc=$?
    if [[ $rc == "$want" ]]; then
        printf '  [ OK ]  %-44s rc=%s\n' "$label" "$rc"
        passed=$((passed + 1))
    else
        printf '  [FAIL]  %-44s rc=%s (want %s), log %s\n' "$label" "$rc" "$want" "$out"
        tail -5 "$out" | sed 's/^/          /'
        failed+=("$label")
    fi
    return 0
}

mapfile -t LABS < <(
    $PY - << 'PY'
import sys
sys.path.insert(0, "tools")
import lab_meta
for n, lab in lab_meta.LABS.items():
    # A lab has a GPU part when any real stage needs CUDA extras or has no CPU path at all
    # (lab1 R is CUDA-only, so `cpu_ok=False` must count, not exclude it).
    gpu = any(st.cuda or not st.cpu_ok for st in lab.stages.values() if not st.guide_only)
    print(n, int(lab.exists()), int(gpu))
PY
)

echo "== environment"
run "check_env --lab all" 0 bash tools/check_env.sh --lab all

for row in "${LABS[@]}"; do
    read -r n exists gpu <<< "$row"
    echo "== lab$n"
    if [[ $exists == 0 ]]; then
        echo "  [ -- ]  lab$n registered in lab_meta but not in this checkout; skipped"
        continue
    fi
    modes=(test)
    ((QUICK)) || modes+=(bench)
    for mode in "${modes[@]}"; do
        case $n in
            2) # Part B needs CUDA; CPU covers A and C host algorithms only.
                [[ $mode == test ]] && run "lab2 reference A test cpu" 0 $PY tools/run.py --lab 2 --impl reference --part A --mode test
                [[ $mode == test ]] && run "lab2 reference C test cpu" 0 $PY tools/run.py --lab 2 --impl reference --part C --mode test
                ;;
            *) run "lab$n reference $mode cpu" 0 $PY tools/run.py --lab "$n" --impl reference --mode "$mode" ;;
        esac
        if [[ $gpu == 1 ]]; then
            run "lab$n reference $mode cuda" 0 $PY tools/run.py --lab "$n" --impl reference --mode "$mode" --device cuda
        fi
    done
    run "lab$n student A (fails at TODO)" 1 $PY tools/run.py --lab "$n" --impl student --part A --mode test
    if [[ -f tools/fault_inject.py ]]; then
        run "lab$n fault_inject cpu" 0 $PY tools/fault_inject.py --lab "$n" --device cpu
        # CUDA-only faults (kernels, streams, R stages) are skipped by a CPU run.
        if [[ $gpu == 1 ]]; then
            run "lab$n fault_inject cuda" 0 $PY tools/fault_inject.py --lab "$n" --device cuda
        fi
    else
        echo "  [ -- ]  fault_inject.py not present yet; skipped"
    fi
done

echo "== repository checks"
run "workload_audit" 0 $PY tools/workload_audit.py --output "$LOG/workload.json"
run "check_links" 0 $PY tools/check_links.py
if [[ -f tools/math_check.py ]]; then
    run "math_check" 0 $PY tools/math_check.py
else
    echo "  [ -- ]  math_check.py not present yet; skipped"
fi
run "format_check" 0 $PY tools/format_check.py
run "test_platform" 0 $PY tools/test_platform.py
for f in tools/*.sh; do run "bash -n $f" 0 bash -n "$f"; done

echo
echo "passed ${passed}, failed ${#failed[@]}; logs in $LOG"
((${#failed[@]} == 0)) || {
    printf '  failed: %s\n' "${failed[@]}"
    exit 1
}
