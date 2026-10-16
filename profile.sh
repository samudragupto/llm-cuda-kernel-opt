#!/usr/bin/env bash
# ==============================================================================
# profile.sh: Nsight Compute (NCU) & Performance Profiling for Add+RMSNorm
# Target: NVIDIA Ada Lovelace (sm_89, RTX 4060 Laptop GPU)
# Metrics:
#   - gpu__time_duration.sum
#   - dram__throughput.avg.pct_of_peak_sustained_elapsed
#   - sm__warps_active.avg.pct_of_peak_sustained_active
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------- Configuration --------------------------------
ARCH="sm_89"
NVCC_FLAGS="-O3 -std=c++17 -arch=${ARCH} -lineinfo --use_fast_math"
SRC="benchmark_rmsnorm.cu"
EXE="./benchmark_rmsnorm.exe"
[ ! -f "$EXE" ] && [ -f "./benchmark_rmsnorm" ] && EXE="./benchmark_rmsnorm"

DEFAULT_TOKENS="16384"
TOKENS="${1:-$DEFAULT_TOKENS}"
CSV_OUT="profile_ncu_results.csv"
NCU_RAW_LOG="ncu_raw.csv"

# ------------------------------- Step 1: Build --------------------------------
echo "======================================================================"
echo " [Step 1/3] Compiling ${SRC} with nvcc (${ARCH})..."
echo "======================================================================"

MAKE_CMD=""
if command -v make &>/dev/null; then
    MAKE_CMD="make"
elif command -v mingw32-make &>/dev/null; then
    MAKE_CMD="mingw32-make"
elif [ -f "/c/MinGW/bin/mingw32-make.exe" ]; then
    MAKE_CMD="/c/MinGW/bin/mingw32-make.exe"
fi

if [ -n "$MAKE_CMD" ] && [ -f "Makefile" ]; then
    $MAKE_CMD all
elif command -v nvcc &>/dev/null; then
    nvcc ${NVCC_FLAGS} ${SRC} -o benchmark_rmsnorm.exe
else
    echo "[WARN] Could not find make or nvcc on PATH, assuming binary exists."
fi

if [ -f "./benchmark_rmsnorm.exe" ]; then
    EXE="./benchmark_rmsnorm.exe"
elif [ -f "./benchmark_rmsnorm" ]; then
    EXE="./benchmark_rmsnorm"
else
    echo "[ERROR] Executable benchmark_rmsnorm not found!"
    exit 1
fi
echo "[OK] Binary ready: ${EXE}"

# -------------------------- Step 2: Nsight Compute ----------------------------
echo ""
echo "======================================================================"
echo " [Step 2/3] Running Nsight Compute (ncu) profiling (tokens=${TOKENS})..."
echo "======================================================================"

METRICS="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

# Find NCU binary
NCU_BIN="ncu"
if ! command -v ncu &>/dev/null; then
    if [ -f "C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.0/ncu.bat" ]; then
        NCU_BIN="C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.0/ncu.bat"
    elif [ -f "/usr/local/cuda/bin/ncu" ]; then
        NCU_BIN="/usr/local/cuda/bin/ncu"
    fi
fi

# Run NCU with CSV output
NCU_SUCCESS=true
rm -f "$NCU_RAW_LOG" "$CSV_OUT"

echo "Executing: ${NCU_BIN} --metrics ${METRICS} ..."
set +e
"${NCU_BIN}" \
    --metrics "${METRICS}" \
    --kernel-name "regex:fused_add_rmsnorm_kernel|rmsnorm_vec_kernel|residual_add_vec_kernel|rmsnorm_naive_kernel|residual_add_naive_kernel" \
    --launch-count 1 \
    --csv \
    --log-file "${NCU_RAW_LOG}" \
    "${EXE}" --tokens "${TOKENS}" --warmup 0 --iters 1 --repeats 1 > ncu_stdout.log 2>&1
NCU_RET=$?
set -e

if grep -q "ERR_NVGPUCTRPERM" ncu_stdout.log 2>/dev/null || [ $NCU_RET -ne 0 ]; then
    NCU_SUCCESS=false
fi

# -------------------------- Step 3: Format & Output ---------------------------
echo ""
echo "======================================================================"
echo " [Step 3/3] Profiling Results & Google Sheets CSV Format"
echo "======================================================================"

if [ "$NCU_SUCCESS" = true ] && [ -f "$NCU_RAW_LOG" ]; then
    echo "[NCU Profiling Succeeded] Extracting metrics from hardware counters..."
    echo ""
    
    python - << 'EOF'
import csv, sys, os

raw_file = "ncu_raw.csv"
if not os.path.exists(raw_file):
    sys.exit(0)

records = {}
with open(raw_file, 'r', encoding='utf-8', errors='ignore') as f:
    # Skip until CSV header starts
    lines = f.readlines()

csv_lines = [l for l in lines if l.startswith('"ID"') or (l.startswith('"') and '","' in l)]
if csv_lines:
    reader = csv.DictReader(csv_lines)
    for row in reader:
        kname = row.get("Kernel Name", "")
        mname = row.get("Metric Name", "")
        mval  = row.get("Metric Value", "")
        munit = row.get("Metric Unit", "")
        if not kname or not mname:
            continue
        
        # Demangle/simplify kernel name
        clean_name = kname
        for prefix in ["fused_add_rmsnorm_kernel", "rmsnorm_vec_kernel", "residual_add_vec_kernel",
                       "rmsnorm_naive_kernel", "residual_add_naive_kernel"]:
            if prefix in kname:
                clean_name = prefix
                break
        
        if clean_name not in records:
            records[clean_name] = {}
        records[clean_name][mname] = (mval, munit)

print(f"{'Kernel Name':<28} | {'Duration':<14} | {'DRAM Throughput':<16} | {'SM Active Warps':<16}")
print("-" * 82)

csv_rows = [["Kernel Name", "Duration (us)", "DRAM Throughput (% Peak)", "SM Active Warps (% Peak)"]]

for kname, metrics in records.items():
    dur_val, dur_unit = metrics.get("gpu__time_duration.sum", ("N/A", ""))
    dram_val, _       = metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", ("N/A", "%"))
    sm_val, _         = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", ("N/A", "%"))
    
    # Format duration to us
    dur_str = f"{dur_val} {dur_unit}"
    dur_us = dur_val
    try:
        fval = float(dur_val.replace(',', ''))
        if "ns" in dur_unit:
            dur_us = f"{fval / 1000.0:.2f}"
            dur_str = f"{fval / 1000.0:.2f} us"
        elif "ms" in dur_unit:
            dur_us = f"{fval * 1000.0:.2f}"
            dur_str = f"{fval * 1000.0:.2f} us"
    except:
        pass
    
    print(f"{kname:<28} | {dur_str:<14} | {dram_val + '%':<16} | {sm_val + '%':<16}")
    csv_rows.append([kname, dur_us, dram_val, sm_val])

print("\n" + "=" * 82)
print(">>> GOOGLE SHEETS COMPATIBLE CSV (Copy lines below):")
print("=" * 82)
for r in csv_rows:
    print(",".join(r))
print("=" * 82)

with open("profile_ncu_results.csv", "w", newline="") as out_f:
    writer = csv.writer(out_f)
    writer.writerows(csv_rows)
print("[OK] Saved to profile_ncu_results.csv")
EOF

else
    # Performance Counter Permission Notification + CUDA Events Fallback
    echo ">> [NOTE: NVIDIA GPU Performance Counter Permissions]"
    echo "   NCU reported: ERR_NVGPUCTRPERM (non-admin hardware counter access restricted)."
    echo "   To enable hardware counters permanently:"
    echo "     * Windows: NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters"
    echo "                Select: 'Allow access to the GPU performance counters to all users'."
    echo "     * Linux:   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"
    echo ""
    echo ">> Extracting high-precision CUDA Events benchmark timings & DRAM throughput..."
    echo ""

    python - "$TOKENS" << 'EOF'
import subprocess, re, sys

# Run the benchmark to extract CUDA Events timings and DRAM throughput
tokens = "16384"
if len(sys.argv) > 1 and sys.argv[1]:
    tokens = sys.argv[1]

cmd = ["./benchmark_rmsnorm.exe", "--tokens", tokens, "--warmup", "20", "--iters", "50", "--repeats", "5"]
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

out = res.stdout
lines = out.splitlines()

# Parse the Performance table
perf_rows = []
in_perf = False
for line in lines:
    if "=== Performance" in line:
        in_perf = True
        continue
    if in_perf and "tokens" in line and "naive" in line:
        continue
    if in_perf and line.strip().startswith("==="):
        break
    if in_perf and "|" in line:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[0].isdigit():
            tok = parts[0]
            times = parts[1].split()
            naive_us = times[0] if len(times) > 0 else "N/A"
            vec_us   = times[1] if len(times) > 1 else "N/A"
            fused_us = times[2] if len(times) > 2 else "N/A"
            
            bw_parts = parts[2].split()
            unfused_bw = bw_parts[0] if len(bw_parts) > 0 else "N/A"
            fused_bw   = bw_parts[1] if len(bw_parts) > 1 else "N/A"
            fused_pct  = re.search(r'\((.*?)\)', parts[2])
            fused_pct_str = fused_pct.group(1) if fused_pct else "N/A"
            
            speedups = parts[3].split()
            x_naive = speedups[0] if len(speedups) > 0 else "N/A"
            x_vec   = speedups[1] if len(speedups) > 1 else "N/A"
            
            perf_rows.append({
                "tokens": tok,
                "naive_us": naive_us,
                "vec_us": vec_us,
                "fused_us": fused_us,
                "unfused_bw": unfused_bw,
                "fused_bw": fused_bw,
                "fused_pct": fused_pct_str,
                "x_naive": x_naive,
                "x_vec": x_vec
            })

print(f"{'Kernel / Variant':<30} | {'Tokens':<8} | {'Time (us)':<12} | {'DRAM BW (GB/s)':<16} | {'% Peak DRAM':<14} | {'Speedup':<10}")
print("-" * 100)

csv_rows = [["Kernel / Variant", "Tokens", "Time (us)", "DRAM Throughput (GB/s)", "% of Peak DRAM Sustained", "Speedup vs Baseline"]]

for r in perf_rows:
    # Naive Baseline
    print(f"{'Baseline (Unfused Naive)':<30} | {r['tokens']:<8} | {r['naive_us']:<12} | {'--':<16} | {'--':<14} | {'1.000x':<10}")
    csv_rows.append(["Baseline (Unfused Naive)", r["tokens"], r["naive_us"], "", "", "1.000x"])
    
    # Vectorized Unfused
    print(f"{'Vectorized Unfused (Ablation)':<30} | {r['tokens']:<8} | {r['vec_us']:<12} | {r['unfused_bw']:<16} | {'--':<14} | {float(r['naive_us'])/float(r['vec_us']):.3f}x")
    csv_rows.append(["Vectorized Unfused (Ablation)", r["tokens"], r["vec_us"], r["unfused_bw"], "", f"{float(r['naive_us'])/float(r['vec_us']):.3f}x"])
    
    # Fully Fused
    print(f"{'Optimized (Fused Add+RMSNorm)':<30} | {r['tokens']:<8} | {r['fused_us']:<12} | {r['fused_bw']:<16} | {r['fused_pct']:<14} | {r['x_naive']:<10}")
    csv_rows.append(["Optimized (Fused Add+RMSNorm)", r["tokens"], r["fused_us"], r["fused_bw"], r["fused_pct"], r["x_naive"]])

print("\n" + "=" * 80)
print(">>> GOOGLE SHEETS COMPATIBLE CSV (Ready to copy-paste into Google Sheets):")
print("=" * 80)
for row in csv_rows:
    print(",".join(f'"{c}"' if ',' in c else str(c) for c in row))
print("=" * 80)

with open("profile_results.csv", "w", newline="") as f:
    for row in csv_rows:
        f.write(",".join(f'"{c}"' if ',' in c else str(c) for c in row) + "\n")
print("[OK] Saved to profile_results.csv")
EOF

fi
