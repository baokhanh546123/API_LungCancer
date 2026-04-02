#!/bin/bash
set -euo pipefail

# Logging
log_info() { echo "[INFO] $1" >&2; }
log_warn() { echo "[WARN] $1" >&2; }
log_error() { echo "[ERROR] $1" >&2; exit 1; }

# Detect OS
detect_os() {
    OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')

    case "$OS_TYPE" in
        darwin*)              OS="macos" ;;
        linux*)               OS="linux" ;;
        *cygwin*|*mingw*|*msys*) OS="windows" ;;
        freebsd*)             OS="freebsd" ;;
        *)                    OS="unknown" ;;
    esac

    echo "$OS"
}

# Detect GPU and tier
detect_gpu() {
  OS=$(detect_os)
  count=0
  total_vram=0
  gpu_names=""
  
  if [ "$OS" = "linux" ] || [ "$OS" = "windows" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
      total_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
      gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd "," -)
    elif command -v rocm-smi >/dev/null 2>&1; then
      count=$(rocm-smi --showproductname | grep -c "GPU")
      total_vram=$(rocm-smi --showmeminfo vram --json | grep total | grep -o '[0-9]*' | awk '{s+=$1} END {print s}')
      gpu_names="amd"
    elif command -v lspci >/dev/null 2>&1 && lspci | grep -qi "intel"; then
      count=1
      total_vram=512
      gpu_names="intel"
    fi
  elif [ "$OS" = "macos" ]; then
    if command -v system_profiler >/dev/null 2>&1; then
      count=$(system_profiler SPDisplaysDataType | grep -c "Chipset Model")
      total_vram=$(system_profiler SPDisplaysDataType | grep "VRAM" | grep -o "[0-9]*" | awk '{s+=$1} END {print s}')
      gpu_names=$(system_profiler SPDisplaysDataType | grep "Chipset Model" | awk -F: '{print $2}' | xargs | sed 's/ /,/g')
      
      # === FIX cho MacBook Pro M4 / M3 / M2 (Apple Silicon) ===
      if [ "${total_vram:-0}" -eq 0 ]; then
        if sysctl hw.optional.arm64 >/dev/null 2>&1; then
          total_vram=$(sysctl -n hw.memsize | awk '{print int($1 / 1048576)}')
        fi
      fi
    elif sysctl hw.optional.arm64 >/dev/null 2>&1; then
      count=1
      total_vram=$(sysctl -n hw.memsize | awk '{print int($1 / 1048576)}')
      gpu_names="apple_silicon"
    fi
  fi

  # Xác định tier (giữ nguyên logic cũ)
  if [ "$total_vram" -ge 8000 ]; then tier="strong"
  elif [ "$total_vram" -ge 4000 ]; then tier="medium"
  elif [ "$total_vram" -gt 0 ]; then tier="weak"
  else tier="none"
  fi

  # === MỚI: Xác định vendor và chipset_name ===
  if [ "$OS" = "macos" ]; then
    vendor="apple"
    chipset_name=$(echo "$gpu_names" | sed 's/^Apple,//' | sed 's/,/ /g' | xargs)
    [ -z "$chipset_name" ] && chipset_name="Apple Silicon"
  elif [[ $gpu_names == *NVIDIA* ]]; then
    vendor="nvidia"
    chipset_name="$gpu_names"
  elif [[ $gpu_names == *AMD* ]] || [ "$gpu_names" = "amd" ]; then
    vendor="amd"
    chipset_name="${gpu_names:-AMD GPU}"
  elif [ "$gpu_names" = "intel" ]; then
    vendor="intel"
    chipset_name="Intel Integrated Graphics"
  else
    vendor="unknown"
    chipset_name="${gpu_names:-unknown}"
  fi

  # Output JSON đầy đủ (đây là thứ Python sẽ đọc)
  echo "{\"os\":\"$OS\",\"gpu_count\":$count,\"gpu_name\":\"$gpu_names\",\"vram_mb\":$total_vram,\"tier\":\"$tier\",\"vendor\":\"$vendor\",\"chipset_name\":\"$chipset_name\"}"
}

# Main
main() {
  os=$(detect_os)
  json=$(detect_gpu)                    # ← JSON đầy đủ

  # Parse để quyết định log + TORCH_DEVICE (giữ nguyên logic cũ)
  tier=$(echo "$json" | grep -o '"tier":"[^"]*"' | cut -d'"' -f4)
  vram=$(echo "$json" | grep -o '"vram_mb":[0-9]*' | cut -d: -f2)

  if [ "$tier" = "none" ] || [ "$vram" -lt 4000 ]; then
    log_info "Weak/no GPU (os=$os,tier=$tier, VRAM=${vram}MB). Fallback to CPU."
    export TORCH_DEVICE="cpu"
  elif [ "$vram" -lt 8000 ]; then
    log_info "Medium GPU (os=$os ,tier=$tier,VRAM=${vram}MB). Performance may be suboptimal."
  else
    log_info "Strong GPU detected."
  fi

  echo "$json"

  exec "$@"
}

main