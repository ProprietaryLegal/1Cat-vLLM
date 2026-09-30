#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/bootstrap_glm53_sm70.sh [--execute] [--venv PATH]
    [--cuda-home PATH]

The default is a command preview. --execute is deliberately explicit because
it creates a virtual environment, installs dependencies, and compiles CUDA
extensions. The GLM SM70 release profile is pinned to CUDA 12.8.93, PyTorch
2.10, and Python 3.12; this script refuses a different nvcc release.
EOF
}

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
venv_dir="${repo_root}/.venv"
cuda_home="${CUDA_HOME:-/opt/toolchains/cuda-12.8.93}"
execute=0

while (($#)); do
    case "$1" in
        --execute)
            execute=1
            ;;
        --venv)
            (($# >= 2)) || { echo "--venv needs a path" >&2; exit 2; }
            venv_dir=$2
            shift
            ;;
        --cuda-home)
            (($# >= 2)) || { echo "--cuda-home needs a path" >&2; exit 2; }
            cuda_home=$2
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

python_bin="${venv_dir}/bin/python"
if [[ -x "${cuda_home}/bin/nvcc" ]]; then
    nvcc_bin="${cuda_home}/bin/nvcc"
    cuda_root="${cuda_home}"
elif [[ -x "${cuda_home}/targets/x86_64-linux/bin/nvcc" ]]; then
    nvcc_bin="${cuda_home}/targets/x86_64-linux/bin/nvcc"
    cuda_root="${cuda_home}/targets/x86_64-linux"
else
    nvcc_bin="${cuda_home}/bin/nvcc"
    cuda_root="${cuda_home}"
fi

if [[ -d "${cuda_home}/targets/x86_64-linux" ]]; then
    cuda_target_root="${cuda_home}/targets/x86_64-linux"
else
    cuda_target_root="${cuda_root}"
fi

cuda_include_paths=(
    "${cuda_root}/include"
    "${cuda_home}/targets/x86_64-linux/include"
)
cuda_library_paths=(
    "${cuda_root}/lib"
    "${cuda_root}/lib64"
    "${cuda_home}/targets/x86_64-linux/lib"
)
nvidia_package_root="${venv_dir}/lib/python3.12/site-packages/nvidia"
cuda_splayed_lib="${venv_dir}/cuda-splayed/lib"
cuda_cudart_library="${cuda_target_root}/lib/libcudart.so"
if [[ ! -f "${cuda_cudart_library}" ]]; then
    cuda_cudart_library="${cuda_root}/lib64/libcudart.so"
fi
cuda_cmake_args="-DCUDA_TOOLKIT_ROOT_DIR=${cuda_target_root}"
cuda_cmake_args+=" -DCUDA_TOOLKIT_TARGET_DIR=${cuda_target_root}"
cuda_cmake_args+=" -DCUDA_NVCC_EXECUTABLE=${nvcc_bin}"
cuda_cmake_args+=" -DCMAKE_LIBRARY_PATH=${cuda_splayed_lib}"
cuda_cmake_args+=" -DCUDAToolkit_INCLUDE_DIR=${nvidia_package_root}/cublas/include"
cuda_cmake_args+=" -DCUDA_CUDART=${cuda_cudart_library}"
cuda_cmake_args+=" -DCUDA_cublas_LIBRARY=${nvidia_package_root}/cublas/lib/libcublas.so.12"
cuda_cmake_args+=" -DCUDA_cublasLt_LIBRARY=${nvidia_package_root}/cublas/lib/libcublasLt.so.12"
cuda_cmake_args+=" -DCUDA_cusparse_LIBRARY=${nvidia_package_root}/cusparse/lib/libcusparse.so.12"
cuda_cmake_args+=" -DCUDA_cusolver_LIBRARY=${nvidia_package_root}/cusolver/lib/libcusolver.so.11"
cuda_cmake_args+=" -DCUDA_nvrtc_LIBRARY=${nvidia_package_root}/cuda_nvrtc/lib/libnvrtc.so.12"
cuda_cmake_args+=" -DCUDA_curand_LIBRARY=${nvidia_package_root}/curand/lib/libcurand.so.10"
cuda_cmake_args+=" -DCUDA_cufft_LIBRARY=${nvidia_package_root}/cufft/lib/libcufft.so.11"
cuda_cmake_args+=" -DCUDA_nvToolsExt_LIBRARY=${nvidia_package_root}/nvtx/lib/libnvToolsExt.so.1"
cuda_include_paths+=(
    "${nvidia_package_root}/cublas/include"
    "${nvidia_package_root}/cuda_cccl/include"
    "${nvidia_package_root}/cuda_runtime/include"
    "${nvidia_package_root}/curand/include"
    "${nvidia_package_root}/cusolver/include"
    "${nvidia_package_root}/cusparse/include"
)
cuda_library_paths+=(
    "${nvidia_package_root}/cublas/lib"
    "${nvidia_package_root}/cuda_runtime/lib"
    "${nvidia_package_root}/cusolver/lib"
    "${nvidia_package_root}/cusparse/lib"
)
cuda_cpath=$(IFS=:; echo "${cuda_include_paths[*]}")
cuda_ldpath=$(IFS=:; echo "${cuda_library_paths[*]}")

build_environment() {
    export CUDA_HOME="${cuda_root}"
    export CUDA_PATH="${cuda_root}"
    export NVCC="${nvcc_bin}"
    export CUDACXX="${nvcc_bin}"
    export CMAKE_CUDA_COMPILER="${nvcc_bin}"
    export CMAKE_ARGS="${cuda_cmake_args}${CMAKE_ARGS:+ ${CMAKE_ARGS}}"
    export PATH="${venv_dir}/bin:${cuda_root}/bin:${cuda_home}/bin:${PATH}"
    export CPATH="${cuda_cpath}${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${cuda_ldpath}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${cuda_ldpath}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TORCH_EXTENSIONS_DIR="${venv_dir}/torch_extensions"
    export TORCH_CUDA_ARCH_LIST=7.0
    export CMAKE_CUDA_ARCHITECTURES=70
    export FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0
    export MAX_JOBS=2
    export CMAKE_BUILD_PARALLEL_LEVEL=2
    export NVCC_THREADS=1
    export PYTHONNOUSERSITE=1
    # vllm/envs.py treats only 1/true as enabled. A wheel-location variable
    # independently enables precompiled mode, so clear it explicitly.
    export VLLM_USE_PRECOMPILED=0
    unset VLLM_PRECOMPILED_WHEEL_LOCATION
}

stage_cuda_library_links() {
    mkdir -p "${cuda_splayed_lib}"
    local name source
    while IFS=' ' read -r name source; do
        [[ -n "${name}" && -f "${source}" ]] || {
            echo "required CUDA runtime library is missing: ${source}" >&2
            exit 1
        }
        ln -sfn "${source}" "${cuda_splayed_lib}/${name}"
    done <<EOF
libcublas.so ${nvidia_package_root}/cublas/lib/libcublas.so.12
libcublasLt.so ${nvidia_package_root}/cublas/lib/libcublasLt.so.12
libcusparse.so ${nvidia_package_root}/cusparse/lib/libcusparse.so.12
libcusolver.so ${nvidia_package_root}/cusolver/lib/libcusolver.so.11
libnvrtc.so ${nvidia_package_root}/cuda_nvrtc/lib/libnvrtc.so.12
libcurand.so ${nvidia_package_root}/curand/lib/libcurand.so.10
libcufft.so ${nvidia_package_root}/cufft/lib/libcufft.so.11
libnvToolsExt.so ${nvidia_package_root}/nvtx/lib/libnvToolsExt.so.1
EOF
}

if (( ! execute )); then
    cat <<EOF
Preview only; no files or packages will be changed.

cd ${repo_root}
uv venv --python 3.12 ${venv_dir}
uv pip install --python ${python_bin} --torch-backend=cu128 \\
    -r requirements/build/cuda.txt
uv pip install --python ${python_bin} --torch-backend=cu128 \\
    -r requirements/cuda.txt
CUDA_HOME=${cuda_root} \\
CUDA_PATH=${cuda_root} \\
NVCC=${nvcc_bin} \\
CUDACXX=${nvcc_bin} \\
CMAKE_CUDA_COMPILER=${nvcc_bin} \\
CMAKE_ARGS="${cuda_cmake_args}" \\
TORCH_CUDA_ARCH_LIST=7.0 \\
CMAKE_CUDA_ARCHITECTURES=70 \\
FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0 \\
MAX_JOBS=2 \\
CMAKE_BUILD_PARALLEL_LEVEL=2 \\
NVCC_THREADS=1 \\
VLLM_USE_PRECOMPILED=0 \\
VLLM_PRECOMPILED_WHEEL_LOCATION= \\
uv pip install --python ${python_bin} --no-build-isolation \\
    --torch-backend=cu128 -e ${repo_root}
EOF
    exit 0
fi

command -v uv >/dev/null || {
    echo "uv is required; install it without using system pip" >&2
    exit 1
}
[[ -x "${nvcc_bin}" ]] || {
    echo "nvcc not found at ${nvcc_bin}" >&2
    exit 1
}

cuda_version=$("${nvcc_bin}" --version | sed -n \
    's/.*release \([0-9][0-9.]*\).*/\1/p' | tail -n 1)
if [[ "${cuda_version}" != 12.8* ]]; then
    cat >&2 <<EOF
Refusing the audited SM70 source build: ${nvcc_bin} reports
CUDA ${cuda_version:-unknown}, while the pinned release profile requires
CUDA 12.8.x.
EOF
    exit 1
fi

if [[ ! -x "${python_bin}" ]]; then
    uv venv --python 3.12 "${venv_dir}"
fi

build_environment
mkdir -p "${TORCH_EXTENSIONS_DIR}"
cd "${repo_root}"
uv pip install --python "${python_bin}" --torch-backend=cu128 \
    -r requirements/build/cuda.txt
uv pip install --python "${python_bin}" --torch-backend=cu128 \
    -r requirements/cuda.txt
stage_cuda_library_links
build_environment

for header in \
    "${nvidia_package_root}/cublas/include/cublas_v2.h" \
    "${nvidia_package_root}/cuda_runtime/include/cuda_runtime.h" \
    "${nvidia_package_root}/curand/include/curand_kernel.h" \
    "${nvidia_package_root}/cusparse/include/cusparse.h"; do
    [[ -f "${header}" ]] || {
        echo "required CUDA development header is missing: ${header}" >&2
        exit 1
    }
done

uv pip install --python "${python_bin}" -r requirements/lint.txt
"${venv_dir}/bin/pre-commit" install

uv pip install --python "${python_bin}" --no-build-isolation \
    --torch-backend=cu128 -e "${repo_root}"

echo "SM70 source build completed in ${venv_dir}"
