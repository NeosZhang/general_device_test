import shutil
import os
import argparse

from utils.process_test import modify_src_code
from utils.utils import sparse_checkout
from utils.unnecessary_tests import unnecessary_tests

# 通过传参设置device_code
parser = argparse.ArgumentParser()
parser.add_argument(
    "device_code",
    type=str,
    help="the supported device_code are 'cuda', 'npu', 'camb', 'muxi', 'ditorch'.",
)
args = parser.parse_args()
device_code = args.device_code

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_TORCH_PATH = MAIN_DIR + "/device_torch/"
ORIGIN_TORCH_PATH = MAIN_DIR + "/origin_torch/"
DEVICE_TEST_PATH = DEVICE_TORCH_PATH + "test"
TORCH_TEST_PATH = ORIGIN_TORCH_PATH + "test"
SHOW_SKIPPED_TESTS = os.getenv("DISPLAY_SKIPPED_TESTS", False)

assert device_code in [
    "cuda",
    "npu",
    "camb",
    "muxi",
    "ditorch",
], "the supported device_code are 'cuda', 'npu', 'camb', 'muxi', 'ditorch'."


if not os.path.exists(ORIGIN_TORCH_PATH):
    sparse_checkout(
        "https://github.com/pytorch/pytorch.git", ORIGIN_TORCH_PATH, ["test"], "v2.1.0"
    )

if device_code == "npu":
    if not os.path.exists(DEVICE_TORCH_PATH):
        sparse_checkout(
            "https://gitee.com/ascend/pytorch.git",
            DEVICE_TORCH_PATH,
            ["test"],
            "v2.1.0",
        )
        from utils.npu_utils.npu_get_synchronized_files import sync_files
        sync_files(ORIGIN_TORCH_PATH, DEVICE_TORCH_PATH)

    # 从device_test_path拷贝测试数据到当前目录
    shutil.copytree(
        DEVICE_TEST_PATH,
        MAIN_DIR + "/tests/device_specified_tests/",
        dirs_exist_ok=True,
    )


# 从torch_test_path拷贝测试脚本，如果对应的脚本在device_test_path已存在或在unnecessary_tests列表中，则跳过
def ignore_tests(dir, contents):
    ignore_list = []
    if os.path.exists(DEVICE_TEST_PATH):
        ignore_list = [
            name
            for name in contents
            if (
                name in unnecessary_tests
                or name in os.listdir(DEVICE_TEST_PATH)
            )
        ]
    else:
        ignore_list = [
            name
            for name in contents
            if (
                name in unnecessary_tests
            )
        ]
    return ignore_list


shutil.copytree(
    TORCH_TEST_PATH,
    MAIN_DIR + "/tests/processed_tests/",
    ignore=ignore_tests,
    dirs_exist_ok=True,
)

# 对processed_tests下的文件进行符号替换处理
os.chdir(MAIN_DIR + "/tests/processed_tests")
for item in os.listdir(os.getcwd()):
    if item.startswith("test_") and item.endswith(".py"):
        modify_src_code(item, device_code)
