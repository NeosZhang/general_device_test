import shutil
import os
import argparse

from utils.process_test import process_src_code
from utils.utils import sparse_checkout
from utils.unnecessary_tests import unnecessary_tests

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_TORCH_PATH = MAIN_DIR + "/origin_torch/"
TORCH_TEST_PATH = ORIGIN_TORCH_PATH + "test"
SHOW_SKIPPED_TESTS = os.getenv("DISPLAY_SKIPPED_TESTS", False)

if not os.path.exists(ORIGIN_TORCH_PATH):
    sparse_checkout(
        "https://github.com/pytorch/pytorch.git", ORIGIN_TORCH_PATH, ["test"], "v2.1.0"
    )


# 从torch_test_path拷贝测试脚本，如果对应的脚本在unnecessary_tests列表中，则跳过
def ignore_tests(dir, contents):
    ignore_list = [name for name in contents if (name in unnecessary_tests)]
    return ignore_list


shutil.copytree(
    TORCH_TEST_PATH,
    MAIN_DIR + "/processed_tests/",
    ignore=ignore_tests,
    dirs_exist_ok=True,
)


os.chdir(MAIN_DIR + "/processed_tests")
for item in os.listdir(os.getcwd()):
    if item.startswith("test_") and item.endswith(".py"):
        process_src_code(item)
