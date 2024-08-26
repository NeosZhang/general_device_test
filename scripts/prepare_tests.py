import shutil
import os
import argparse
import subprocess

from process_test_utils import modify_src_code

# 通过传参设置device_code
parser = argparse.ArgumentParser()
parser.add_argument(
    "device_code",
    type=str,
    help="the supported device_code are 'cuda', 'npu', 'camb', 'muxi'.",
)
args = parser.parse_args()
device_code = args.device_code

assert device_code in [
    "cuda",
    "npu",
    "camb",
    "muxi",
], "the supported device_code are 'cuda', 'npu', 'camb', 'muxi'."


def sparse_checkout(repo_url, destination, paths, branch="main", depth=1):
    destination = os.path.abspath(
        destination
    )  # Ensure the destination is an absolute path

    try:
        # Perform a shallow clone with the specified depth and branch
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--depth",
                str(depth),
                "--single-branch",
                "-b",
                branch,
                repo_url,
                destination,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git clone failed: {e}")

    git_dir = os.path.join(destination, ".git")

    try:
        # Initialize sparse checkout and set the specified paths
        subprocess.run(
            [
                "git",
                "--git-dir",
                git_dir,
                "--work-tree",
                destination,
                "sparse-checkout",
                "init",
                "--cone",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "--git-dir",
                git_dir,
                "--work-tree",
                destination,
                "sparse-checkout",
                "set",
            ]
            + paths,
            check=True,
        )
        # Checkout the files in the sparse checkout
        subprocess.run(
            ["git", "--git-dir", git_dir, "--work-tree", destination, "checkout"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Sparse checkout failed: {e}")

    print("Sparse checkout 完成!")


device_torch_path = "./device_torch/"
origin_torch_path = "./origin_torch/"

if not os.path.exists(origin_torch_path):
    sparse_checkout(
        "https://github.com/pytorch/pytorch.git", origin_torch_path, ["test"], "v2.1.0"
    )

if device_code == "npu":
    if not os.path.exists(device_torch_path):
        sparse_checkout(
            "https://gitee.com/ascend/pytorch.git",
            device_torch_path,
            ["test"],
            "v2.1.0",
        )


device_test_path = device_torch_path + "test"
torch_test_path = origin_torch_path + "test"

# torch/test中的某些测例对于国产设备是不必要的，或者对应的测试脚本不适用于国产设备
# 所以使用该脚本进行过滤，对过滤后的文件进行处理
unnecessary_tests = [
    "_nvfuser",
    "_test_bazel.py",
    "ao",
    "ao/sparsity",
    "backends",
    "benchmark_utils",
    "cpp",
    "cpp_api_parity",
    "cpp_extensions",
    "custom_backend",
    "custom_operator",
    "distributed",
    "distributions",
    "dynamo",
    "edge",
    "error_messages",
    "export",
    "forward_backward_compatibility",
    "functorch",
    "fx",
    "inductor",
    "jit",
    "jit_hooks",
    "lazy",
    "mobile",
    "onnx",
    "onnx_caffe2",
    "package",
    "profiler",
    "quantization",
    "scripts",
    "test_img",
    "torch_np",
    "typing",
    "test_ao_sparsity.py",
    "test_autograd_fallback.py",
    "test_bundled_images.py",
    "test_bundled_inputs.py",
    "test_comparison_utils.py",
    "test_compile_benchmark_util.py",
    "test_content_store.py",
    "test_cpp_extensions_aot.py",
    "test_cpp_extensions_jit.py",
    "test_cpp_extensions_open_device_registration.py",
    "test_cuda_expandable_segments.py",
    "test_cuda_multigpu.py",
    "test_cuda_nvml_based_avail.py",
    "test_cuda_primary_ctx.py",
    "test_cuda_sanitizer.py",
    "test_cuda_trace.py",
    "test_cuda.py",
    "test_custom_ops.py",
    "test_dataloader.py",
    "test_decomp.py",
    "test_deploy.py",
    "test_determination.py",
    "test_dispatch.py",
    "test_dlpack.py",
    "test_expanded_weights.py",
    "test_foreach.py",
    "test_function_schema.py",
    "test_functional_autograd_benchmark.py",
    "test_functional_optim.py",
    "test_functionalization_of_rng_ops.py",
    "test_functionalization.py",
    "test_futures.py",
    "test_fx_experimental.py",
    "test_fx_passes.py",
    "test_fx_reinplace_pass.py",
    "test_fx.py",
    "test_hub.py",
    "test_itt.py",
    "test_jit_autocast.py",
    "test_jit_cuda_fuser.py",
    "test_jit_disabled.py",
    "test_jit_fuser_legacy.py",
    "test_jit_fuser_te.py",
    "test_jit_fuser.py",
    "test_jit_legacy.py",
    "test_jit_llga_fuser.py",
    "test_jit_profiling.py",
    "test_jit_simple.py",
    "test_jit_string.py",
    "test_jit.py",
    "test_jiterator.py",
    "test_kernel_launch_checks.py",
    "test_license.py",
    "test_linalg.py",
    "test_logging.py",
    "test_masked.py",
    "test_maskedtensor.py",
    "test_matmul_cuda.py",
    "test_meta.py",
    "test_metal.py",
    "test_mkl_verbose.py",
    "test_mkldnn_fusion.py",
    "test_mkldnn_verbose.py",
    "test_mkldnn.py",
    "test_mobile_optimizer.py",
    "test_model_dump.py",
    "test_module_init.py",
    "test_modules.py",
    "test_monitor.py",
    "test_mps.py",
    "test_multiprocessing_spawn.py",
    "test_multiprocessing.py",
    "test_namedtuple_return_api.py",
    "test_native_functions.py",
    "test_native_mha.py",
    "test_nestedtensor.py",
    "test_nnapi.py",
    "test_numba_integration.py",
    "test_numpy_interop.py",
    "test_nvfuser_frontend.py",
    "test_openmp.py",
    "test_ops_fwd_gradients.py",
    "test_ops_gradients.py",
    "test_ops_jit.py",
    "test_ops.py",
    "test_out_dtype_op.py",
    "test_overrides.py",
    "test_package.py",
    "test_per_overload_api.py",
    "test_pruning_op.py",
    "test_public_bindings.py",
    "test_python_dispatch.py",
    "test_pytree.py",
    "test_quantization.py",
    "test_segment_reductions.py",
    "test_set_default_mobile_cpu_allocator.py",
    "test_show_pickle.py",
    "test_sparse_semi_structured.py",
    "test_sparse.py",
    "test_static_runtime.py",
    "test_subclass.py",
    "test_sympy_utils.py",
    "test_tensor_creation_ops.py",
    "test_tensorexpr_pybind.py",
    "test_throughput_benchmark.py",
    "test_torch.py",
    "test_transformers.py",
    "test_type_hints.py",
    "test_type_info.py",
    "test_type_promotion.py",
    "test_typing.py",
    "test_unary_ufuncs.py",
    "test_utils.py",
    "test_view_ops.py",
    "test_vulkan.py",
    "test_weak.py",
    "test_xnnpack_integration.py",
]

if device_code == "npu":
    # 1. 从device_test_path拷贝测试数据到当前目录
    shutil.copytree(device_test_path, "../modified_tests/", dirs_exist_ok=True)


# 2. 从torch_test_path拷贝测试脚本，如果对应的脚本在device_test_path已存在或在unnecessary_tests列表中，则跳过
def ignore_tests(dir, contents):
    ignore_list = []
    if os.path.exists(device_test_path):
        ignore_list = [
            name
            for name in contents
            if (
                name in unnecessary_tests
                or name in os.listdir(device_test_path)
                or ("." in name and "test_" not in name)
            )
        ]
    else:
        ignore_list = [
            name
            for name in contents
            if (name in unnecessary_tests or ("." in name and "test_" not in name))
        ]
    return ignore_list


shutil.copytree(
    torch_test_path, "../to_process_tests/", ignore=ignore_tests, dirs_exist_ok=True
)

# 3. 对to_process_tests下的文件进行符号替换处理
os.chdir("../to_process_tests")
for item in os.listdir(os.getcwd()):
    if item.startswith("test_") and item.endswith(".py"):
        modify_src_code(item, device_code)
