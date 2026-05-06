# the OPS = sparse OPS/2

hardware_params = {
    # NOTICES: For GPU, we use Register File Size as on-chip buffer size
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 154.8e12, "INT8": 309.7e12, "onchip_buffer": 21504e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000_Ada": {"bandwidth": 960e9, "FP16": 364.2e12, "INT8": 728.5e12, "onchip_buffer": 36352e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},  # use 40G data
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A800_80G_SXM": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12, "INT8": 299.3e12, "onchip_buffer": 21504e3},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },  # use SXM data
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12 / 2, "INT8": 3026e12 / 2, "onchip_buffer": 29184e3},
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "onchip_buffer": 36352e3},
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_13900k": {"bandwidth": 89.6e9, "FP16": 8 * 5.4e9 * (512 / 16), "onchip_buffer": 36e6},
    # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
    # Blackwell maintains 256 KB Register File per SM (same as Hopper)
    "nvidia_B200": {"bandwidth": 8e12, "FP16": 2.25e15, "INT8": 4.5e15, "onchip_buffer": 37888e3},
    "nvidia_GB200": {"bandwidth": 8e12, "FP16": 2.5e15, "INT8": 5.0e15, "onchip_buffer": 37888e3},
    "nvidia_B100": {"bandwidth": 8e12, "FP16": 1.75e15, "INT8": 3.5e15, "onchip_buffer": 30720e3},
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
    # GA102 SM has 256 KB Register File
    "nvidia_RTX_3090": {"bandwidth": 936e9, "FP16": 71.16e12, "INT8": 284e12, "onchip_buffer": 20992e3},
    "nvidia_RTX_3090_Ti": {"bandwidth": 1008e9, "FP16": 160e12, "INT8": 320e12, "onchip_buffer": 21504e3},
    # https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    # Ada SM has 256 KB Register File
    "nvidia_RTX_4080": {"bandwidth": 716.8e9, "FP16": 97.47e12, "INT8": 389.9e12, "onchip_buffer": 19456e3},
    "nvidia_RTX_4090": {"bandwidth": 1008e9, "FP16": 165.2e12, "INT8": 660.6e12, "onchip_buffer": 32768e3},
    # https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
    # GB203/GB202 SM has 256 KB Register File
    "nvidia_RTX_5080": {"bandwidth": 960e9, "FP16": 112.6e12, "INT8": 450.2e12, "onchip_buffer": 21504e3},
    "nvidia_RTX_5090": {"bandwidth": 1792e9, "FP16": 209.5e12, "INT8": 838e12, "onchip_buffer": 43520e3},
    # https://www.nvidia.com/en-us/data-center/l40s/
    "nvidia_L40S": {"bandwidth": 864e9, "FP16": 362e12, "INT8": 733e12, "onchip_buffer": 36352e3},
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
    "nvidia_A10": {"bandwidth": 600e9, "FP16": 62.5e12, "INT8": 125e12, "onchip_buffer": 18432e3},
    # https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    "nvidia_L4": {"bandwidth": 300e9, "FP16": 121e12, "INT8": 242e12, "onchip_buffer": 14848e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    "nvidia_A30": {"bandwidth": 933e9, "FP16": 165e12, "INT8": 330e12, "onchip_buffer": 14336e3},
    # https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu
    # Same GH100 die as H100, only memory subsystem upgraded to HBM3e
    "nvidia_H200": {"bandwidth": 4.8e12, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    # https://www.amd.com/en/products/accelerators/instinct/mi300.html
    # AMD GPUs use L2 Cache as onchip_buffer proxy (no direct RF equivalent)
    "amd_MI300X": {"bandwidth": 5.3e12, "FP16": 1.305e15, "INT8": 2.61e15, "onchip_buffer": 32768e3},
    # https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
    "amd_MI250X": {"bandwidth": 3.277e12, "FP16": 383e12, "INT8": 383e12, "onchip_buffer": 16384e3},
    # https://www.amd.com/en/products/accelerators/instinct/mi100.html
    "amd_MI100": {"bandwidth": 1.23e12, "FP16": 184.6e12, "INT8": 92.3e12, "onchip_buffer": 8192e3},
}
