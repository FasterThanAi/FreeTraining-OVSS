#!/usr/bin/env bash
# ============================================================
# SAM 3 lab-PC readiness check
# Run this FIRST, before installing anything.
#   bash check_env.sh
# Everything here is read-only. It installs nothing.
# ============================================================

echo "=========================================="
echo " SAM 3 environment check"
echo " host: $(hostname)   user: $(whoami)   date: $(date)"
echo "=========================================="

echo
echo "--- 1. OS ---"
[ -f /etc/os-release ] && grep PRETTY_NAME /etc/os-release || uname -a
echo "kernel: $(uname -r)   arch: $(uname -m)"

echo
echo "--- 2. GPU + DRIVER  (the make-or-break check) ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version \
               --format=csv,noheader 2>/dev/null \
      || nvidia-smi
    echo
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    echo "Driver's max supported CUDA: ${CUDA_VER:-UNKNOWN}"
    if [ -n "$CUDA_VER" ]; then
        MAJ=${CUDA_VER%%.*}; MIN=${CUDA_VER##*.}
        if [ "$MAJ" -gt 12 ] || { [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; }; then
            echo ">>> OK. Meets SAM 3's CUDA 12.6+ requirement."
        elif [ "$MAJ" -eq 12 ]; then
            echo ">>> MARGINAL. CUDA 12.0-12.5. cu128 wheels usually still run via"
            echo "    CUDA minor-version compatibility, but this is below spec."
            echo "    Try it; if you hit driver errors you need an admin driver upgrade."
        else
            echo ">>> BLOCKER. Below CUDA 12. You need a driver upgrade (requires root)."
            echo "    Email your lab admin now - this is the long pole."
        fi
    fi
else
    echo ">>> nvidia-smi not found. Either no NVIDIA GPU, or drivers are not installed."
    echo "    Check for hardware:"
    lspci 2>/dev/null | grep -i -E 'vga|3d|nvidia' || echo "    (lspci unavailable)"
fi

echo
echo "--- 3. Other GPU users (shared machine?) ---"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -v '^$')
    if [ -z "$PROCS" ]; then echo "GPU is idle - you have it to yourself right now."
    else echo "$PROCS"; echo ">>> Someone else is using the GPU. Coordinate, and set CUDA_VISIBLE_DEVICES."; fi
else
    echo "(skipped - driver not responding)"
fi

echo
echo "--- 4. Python ---"
for p in python3 python3.12 python3.11 python3.10; do
    command -v $p &>/dev/null && echo "$p -> $($p --version 2>&1)"
done
echo "SAM 3 needs Python 3.12+. If absent, conda supplies it - no root needed."

echo
echo "--- 5. Conda ---"
if command -v conda &>/dev/null; then
    echo "conda: $(conda --version)  at $(command -v conda)"
    echo ">>> Good, use it."
else
    echo ">>> No conda. Install Miniconda into \$HOME (no root required) - see SETUP_SAM3.md step 2."
fi

echo
echo "--- 6. Disk space ---"
echo "HOME ($HOME):"; df -h "$HOME" | tail -1
for d in /scratch /data /mnt/scratch /tmp; do
    [ -d "$d" ] && [ -w "$d" ] && { echo "$d (writable):"; df -h "$d" | tail -1; }
done
echo "Home quota (if enforced):"; quota -s 2>/dev/null || echo "  (no quota command / not enforced)"
echo ">>> Need ~40GB: conda env ~15GB, checkpoints ~5GB, datasets ~20GB+."
echo "    If HOME is small or quota'd, point conda + HF_HOME at a scratch disk."

echo
echo "--- 7. Root access ---"
if sudo -n true 2>/dev/null; then echo ">>> You have passwordless sudo."
else echo ">>> No passwordless sudo (normal for lab PCs). Fine - nothing below needs root."; fi

echo
echo "--- 8. Network reachability ---"
for url in https://github.com https://pypi.org https://huggingface.co https://download.pytorch.org; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 10 "$url" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "301" ] || [ "$code" = "302" ]; then
        echo "  OK      $url"
    else
        echo "  BLOCKED $url  (HTTP $code) <-- may need a proxy"
    fi
done
[ -n "$http_proxy$https_proxy" ] && echo "  proxy env is set: ${https_proxy:-$http_proxy}"

echo
echo "--- 9. Build tools (needed for pip install -e .) ---"
for t in git gcc g++ make cmake; do
    command -v $t &>/dev/null && echo "  OK      $t  ($($t --version 2>&1 | head -1))" || echo "  MISSING $t"
done
echo "  (missing ones can come from conda: conda install -c conda-forge gcc_linux-64 gxx_linux-64 make cmake)"

echo
echo "=========================================="
echo " Send this whole output to Claude and it can tell you exactly"
echo " which install path to take."
echo "=========================================="
