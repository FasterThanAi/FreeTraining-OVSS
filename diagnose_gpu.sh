#!/usr/bin/env bash
# ============================================================
# NVIDIA driver diagnosis
#   nvidia-smi exists but can't talk to the driver.
#   This narrows down WHY. Read-only, installs nothing.
#   bash diagnose_gpu.sh
# ============================================================

echo "=========================================="
echo " NVIDIA driver diagnosis"
echo " kernel: $(uname -r)"
echo "=========================================="

echo
echo "--- A. Is there physically an NVIDIA GPU? ---"
if command -v lspci &>/dev/null; then
    GPUS=$(lspci | grep -i -E 'nvidia|vga|3d|display')
    echo "${GPUS:-  (nothing matched)}"
    echo "$GPUS" | grep -qi nvidia \
        && echo ">>> YES - NVIDIA hardware present. This is a software problem, and fixable." \
        || echo ">>> NO NVIDIA GPU DETECTED. Check with your lab admin what this box actually has."
else
    echo "lspci missing; trying sysfs:"
    ls /sys/bus/pci/devices/*/vendor 2>/dev/null | while read f; do
        [ "$(cat $f)" = "0x10de" ] && echo "  NVIDIA device at $(dirname $f)"
    done
fi

echo
echo "--- B. Is the kernel module loaded? ---"
LSMOD=$(lsmod | grep -E '^nvidia|^nouveau')
if [ -n "$LSMOD" ]; then echo "$LSMOD"; else echo "  (no nvidia or nouveau module loaded)"; fi
echo "$LSMOD" | grep -q '^nvidia' \
    && echo ">>> nvidia module IS loaded - the problem is elsewhere (see D, E)." \
    || echo ">>> nvidia module NOT loaded. This is almost certainly your problem."
echo "$LSMOD" | grep -q '^nouveau' \
    && echo ">>> WARNING: open-source 'nouveau' driver is loaded. It conflicts with the NVIDIA driver and must be blacklisted."

echo
echo "--- C. Driver version reported by kernel ---"
if [ -f /proc/driver/nvidia/version ]; then cat /proc/driver/nvidia/version
else echo "  /proc/driver/nvidia/version absent -> driver definitely not running."; fi

echo
echo "--- D. Which driver packages are installed? ---"
dpkg -l 2>/dev/null | grep -E 'nvidia-driver|nvidia-dkms|nvidia-utils|cuda-drivers' \
    | awk '{print "  "$1"  "$2"  "$3}' || echo "  (none found)"

echo
echo "--- E. DKMS: is the module BUILT for the running kernel? ---"
if command -v dkms &>/dev/null; then
    dkms status 2>/dev/null || echo "  (dkms status returned nothing)"
    echo
    echo "  running kernel: $(uname -r)"
    echo "  If the nvidia entry above does not list this exact kernel version,"
    echo "  the module was never rebuilt after a kernel upgrade. That is THE most"
    echo "  common cause of this exact error."
else
    echo "  dkms not installed"
fi

echo
echo "--- F. Is the module file present for this kernel? ---"
# Match the real proprietary driver only. nvidiafb.ko / nvidia_cspmu.ko are
# unrelated in-tree kernel modules and must NOT be counted.
REAL='updates/dkms/nvidia|kernel/nvidia|nvidia-current|/nvidia\.ko|nvidia_drm|nvidia_uvm|nvidia_modeset'
find /lib/modules/$(uname -r) -name 'nvidia*.ko*' 2>/dev/null | grep -E "$REAL" | head -5 \
    || echo "  none found for $(uname -r)  <-- driver not built for the running kernel"
echo "  Real driver modules present for OTHER installed kernels:"
for k in $(ls /lib/modules/ 2>/dev/null); do
    n=$(find /lib/modules/$k -name 'nvidia*.ko*' 2>/dev/null | grep -cE "$REAL")
    [ "$n" -gt 0 ] && echo "    $k  -> $n module(s)  <-- BOOTABLE FALLBACK: pick this kernel in GRUB"
done

echo
echo "--- G. Secure Boot (blocks unsigned kernel modules) ---"
if command -v mokutil &>/dev/null; then
    SB=$(mokutil --sb-state 2>/dev/null); echo "  $SB"
    echo "$SB" | grep -qi enabled && echo ">>> Secure Boot ENABLED - can silently block the NVIDIA module. Common cause."
else
    [ -d /sys/firmware/efi ] && echo "  EFI system, mokutil unavailable - ask admin about Secure Boot" \
                            || echo "  Legacy BIOS - Secure Boot not applicable"
fi

echo
echo "--- H. Recent kernel complaints about NVIDIA ---"
dmesg 2>/dev/null | grep -i -E 'nvidia|nouveau|NVRM' | tail -15 \
    || echo "  (dmesg restricted - try: sudo dmesg | grep -i nvidia)"

echo
echo "--- I. Installed kernels (for a GRUB fallback boot) ---"
ls /boot/vmlinuz-* 2>/dev/null | sed 's|/boot/vmlinuz-|  |'
echo "  currently running: $(uname -r)"

echo
echo "--- J. Do you have sudo at all (with password)? ---"
if sudo -n true 2>/dev/null; then echo "  passwordless sudo: YES"
else
    groups | tr ' ' '\n' | grep -qE '^(sudo|admin|wheel)$' \
      && echo "  You ARE in the sudo group -> you can sudo WITH a password. You can fix this yourself." \
      || echo "  Not in sudo group -> you need your lab admin."
fi
echo "  your groups: $(groups)"

echo
echo "=========================================="
echo " Send this output to Claude for the exact fix."
echo "=========================================="
