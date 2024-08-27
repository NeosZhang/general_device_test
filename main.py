import shutil
import os
import argparse

from utils.process_test import modify_src_code, display_skipped_tests
from utils.utils import sparse_checkout
from utils.unnecessary_tests import unnecessary_tests

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

main_dir = os.path.dirname(os.path.abspath(__file__))
print(main_dir)
device_torch_path = main_dir + "/device_torch/"
origin_torch_path = main_dir + "/origin_torch/"

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
show_skipped_tests = os.getenv("DISPLAY_SKIPPED_TESTS", False)


if device_code == "npu":
    # 1. 从device_test_path拷贝测试数据到当前目录
    shutil.copytree(
        device_test_path, main_dir + "/tests/device_specified_tests/", dirs_exist_ok=True
    )

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
    torch_test_path,
    main_dir + "/tests/processed_tests/",
    ignore=ignore_tests,
    dirs_exist_ok=True,
)

# 3. 对processed_tests下的文件进行符号替换处理
os.chdir(main_dir + "/tests/processed_tests")
for item in os.listdir(os.getcwd()):
    if item.startswith("test_") and item.endswith(".py"):
        modify_src_code(item, device_code)

if show_skipped_tests == "True":
    display_skipped_tests(test_directory=main_dir + "/tests/")
