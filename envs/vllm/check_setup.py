#!/usr/bin/env python3
import platform
import subprocess
import sys


def check_cmd(cmd):
    """Run a shell command and return (success, stdout)"""
    try:
        out = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, text=True
        )
        return True, out.strip()
    except subprocess.CalledProcessError as e:
        return False, e.output.strip()


def print_section(title):
    print("\n" + "=" * 70)
    print(f"[{title}]")
    print("=" * 70)


def main():
    print_section("Python 基本信息")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"平台: {platform.system()} {platform.release()} ({platform.machine()})")

    # ---------------- Torch 检查 ----------------
    print_section("PyTorch 检查")
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 构建版本: {torch.version.cuda}")  # type: ignore
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"检测到 {device_count} 个 CUDA 设备:")
            for i in range(device_count):
                print(f"  - GPU{i}: {torch.cuda.get_device_name(i)}")
                print(f"    Capability: {torch.cuda.get_device_capability(i)}")

            # 驱动版本与CUDA runtime
            ok, smi_out = check_cmd(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
            )
            if ok:
                driver_ver = smi_out.splitlines()[0].strip()
                print(f"系统 NVIDIA 驱动版本: {driver_ver}")
            else:
                print("⚠️ 无法获取驱动版本，请确认是否安装了 NVIDIA 驱动。")

            # 检查 torch CUDA 版本与驱动是否兼容
            if torch.version.cuda:  # type: ignore
                cuda_major = int(
                    torch.version.cuda.split(".")[0]  # type: ignore
                )
                driver_ok, driver_str = check_cmd(
                    "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
                )
                if driver_ok:
                    drv = driver_str.split(".")[0]
                    print(f"驱动版本检测结果: {driver_str}")
                    print("兼容性提示：")
                    print(
                        f"  - PyTorch 构建基于"
                        f" CUDA {torch.version.cuda}"  # type: ignore
                    )
                    print("  - 如果你的驱动版本低于该 CUDA 要求，可能会报错。")
                    print(
                        "  - 可查阅 https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/"
                    )
                else:
                    print("⚠️ 无法检查驱动兼容性（未检测到 nvidia-smi）")
        else:
            print("未检测到可用 GPU，系统可能是 CPU-only 环境。")

    except ImportError as e:
        print(f"❌ 未能导入 torch: {e}")
        print("请检查是否已安装 PyTorch。")

    # ---------------- vLLM 检查 ----------------
    print_section("vLLM 检查")
    try:
        import vllm

        print(f"vLLM 版本: {getattr(vllm, '__version__', '未知')}")
        print("✅ vLLM 模块导入成功。")
    except ImportError as e:
        print(f"❌ 未能导入 vLLM: {e}")
        print("请检查是否在当前环境中安装了 vLLM。")

    # ---------------- 附加 GPU 工具检查 ----------------
    print_section("GPU 工具检测")
    ok, nvcc_out = check_cmd("nvcc --version")
    if ok:
        print("检测到系统已安装 CUDA Toolkit：")
        print(nvcc_out)
    else:
        print(
            "未检测到 nvcc，可推测未安装系统级 CUDA Toolkit（这在使用预编译 wheel 时是正常的）。"
        )

    ok, smi_out = check_cmd("nvidia-smi -L")
    if ok:
        print("nvidia-smi 输出：")
        print(smi_out)
    else:
        print("⚠️ 无法运行 nvidia-smi，可能是驱动未安装或路径未配置。")

    print_section("结论")
    print("✅ 如果上方所有模块均能导入，且 GPU/驱动信息正常，即可认为环境安装成功。")
    print("⚠️ 若 CUDA 构建版本与驱动不匹配，请升级或降级对应组件。")


if __name__ == "__main__":
    main()
