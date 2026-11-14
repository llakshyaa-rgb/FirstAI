#!/data/data/com.termux/files/usr/bin/bash
set -u
echo "=== Android GPU capability probe ==="
echo "Device:"
getprop ro.product.model 2>/dev/null || true
echo "GPU renderer (if accessible):"
for f in /sys/class/kgsl/kgsl-3d0/gpu_model /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies; do
  [ -r "$f" ] && { echo "--- $f"; cat "$f"; }
done
echo
echo "Vulkan loader/layers:"
if command -v vulkaninfo >/dev/null 2>&1; then vulkaninfo --summary 2>&1 | head -n 80
else echo "vulkaninfo is not installed in this Termux environment."; fi
echo
echo "OpenCL:"
if command -v clinfo >/dev/null 2>&1; then clinfo 2>&1 | head -n 80
else echo "clinfo is not installed."; fi
echo
echo "Important: Debian proot's PyTorch wheel is CPU-oriented. Android GPU access requires a compatible Android-side Vulkan/OpenCL/QNN runtime and a model backend built for it. A detected Adreno GPU alone does not make this FirstGPT PyTorch model GPU-accelerated."
