import os
import re


def split_script(input_file):
    """split_script用于将测试脚本test_xxx.py拆分为import部分和代码主体部分"""
    # 用于匹配 import 和 from ... import 语句（包括多行）
    import_pattern = re.compile(r"^\s*(import\s+\w+|from\s+\w+(\.\w+)*\s+import\s+)")
    continuation_pattern = re.compile(r"^\s*[\(\,]")

    # 读取原始脚本
    with open(input_file, "r") as f:
        lines = f.readlines()

    # 搜索import代码的范围，从代码开头到一个等号为止
    search_import_lines = []
    for line in lines:
        if "=" in line:
            break
        search_import_lines.append(line)

    split_line = 0
    is_import = False
    for i in range(len(search_import_lines)):
        if search_import_lines[i] == "":
            continue
        # 检查是否是 import 语句的开始
        if import_pattern.match(search_import_lines[i]):
            is_import = True  # 标记为 import 语句
            split_line = i
        # 检查是否是多行 import 语句的继续部分
        elif is_import or (continuation_pattern.match(search_import_lines[i])):
            # 检查是否结束了 import 语句
            if ")" in search_import_lines[i]:
                is_import = False
                split_line = i
        else:
            is_import = False  # 结束 import 语句标记

    # 将结果返回为字符串
    imports_text = "".join(lines[: split_line + 1])
    functions_text = "".join(lines[split_line + 1 :])
    return imports_text, functions_text


def device_patch_for_test(device_code: str):
    import_torch = ""
    test_code_map = {
        "onlyCUDAAndPRIVATEUSE1": "",
        "TEST_MULTIGPU": "",
        "TEST_CUDA_GRAPH": "",
        "TEST_CUDA": "",
        "TEST_CUDNN": "",
        "torch.cuda": "",
        "onlyCUDA": "",
        "RUN_CUDA": "",
        "dtypesIfCUDA": "",
        ".cuda(": "",
        '"cuda"': "",
        "'cuda'": "",
        "cuda:": "",
    }
    if device_code == "cuda":
        pass
    elif device_code == "npu":
        import_torch = """import torch_npu
import torch_npu.testing
from torch.testing._internal.common_utils import TEST_PRIVATEUSE1
from torch.testing._internal.common_device_type import onlyPRIVATEUSE1, dtypesIfPRIVATEUSE1
TEST_MULTINPU = TEST_PRIVATEUSE1 and torch_npu.npu.device_count() >= 2
RUN_PRIVATEUSE1_MULTI_GPU = TEST_MULTINPU
TEST_PRIVATEUSE1_VERSION = 5130
RUN_PRIVATEUSE1 = True
TEST_NPU = torch_npu.npu.is_available()
TEST_BFLOAT16 = TEST_NPU and torch_npu.npu.is_bf16_supported()
RUN_PRIVATEUSE1_HALF = RUN_PRIVATEUSE1"""

        test_code_map["onlyCUDAAndPRIVATEUSE1"] = "onlyPRIVATEUSE1"
        test_code_map["TEST_MULTIGPU"] = "TEST_MULTINPU"
        test_code_map["TEST_CUDA_GRAPH"] = "TEST_PRIVATEUSE1"
        test_code_map["TEST_CUDA"] = "TEST_PRIVATEUSE1"
        test_code_map["TEST_CUDNN"] = "TEST_PRIVATEUSE1"
        test_code_map["torch.cuda"] = "torch_npu.npu"
        test_code_map["onlyCUDA"] = "onlyPRIVATEUSE1"
        test_code_map["RUN_CUDA"] = "RUN_PRIVATEUSE1"
        test_code_map["dtypesIfCUDA"] = "dtypesIfPRIVATEUSE1"
        test_code_map[".cuda("] = ".npu("
        test_code_map['"cuda"'] = '"npu"'
        test_code_map["'cuda'"] = "'npu'"
        test_code_map["cuda:"] = "npu:"
    elif device_code == "muxi":
        pass
    else:
        raise ValueError(
            "invald device code! The legal device code are: cuda, npu, muxi."
        )
    return import_torch, test_code_map


def replace_in_text(text, mapping):
    # 将映射关系字典的键转换为正则表达式可以接受的格式，确保能正确处理特殊字符
    pattern = re.compile("|".join(re.escape(key) for key in mapping.keys()))
    # 使用正则表达式替换文本中的键
    return pattern.sub(lambda x: mapping[x.group()], text)


def modify_src_code(src: str, device_code: str):
    code_patch = device_patch_for_test(device_code)
    imports_text, functions_text = split_script(src)

    imports_text = code_patch[0] + imports_text
    functions_text = replace_in_text(functions_text, code_patch[1])

    with open(src, "w", encoding="utf-8") as file:
        file.write(imports_text + functions_text)


def display_skipped_tests(test_directory):
    """
    Description:
        Show skipped tests and their reasons.
    
    Args:
        test_directory (str): The root path contains all test files.
    
    Warning:
        This function should use after modify_src_code func, and 
        work with DISPLAY_SKIPPED_TESTS enviroment variables.
    """
    import_unittest_module_code = """
import unittest
"""

    custom_test_runner_code = """
class CustomTextTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_tests = []

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.skipped_tests.append((test, reason))

    def printErrors(self):
        super().printErrors()

        if self.skipped_tests:
            self.stream.writeln(self.separator1)
            for test, reason in self.skipped_tests:
                self.stream.writeln(f"Skip {self.getDescription(test)}: {reason}")

class CustomTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTextTestResult(self.stream, self.descriptions, self.verbosity)

"""
    # 遍历目录，找到所有以test_开头的.py文件
    for root, dirs, files in os.walk(test_directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "import unittest\n" not in content:
                    content = import_unittest_module_code + content
                # 检查是否已经包含CustomTextTestRunner代码，避免重复添加
                if "CustomTextTestRunner" not in content:
                    match = re.search(r"(import unittest\n)", content)
                    if match:
                        # 在 import unittest 后插入自定义测试运行器代码
                        insert_position = match.end()
                        content = (
                            content[:insert_position]
                            + custom_test_runner_code
                            + content[insert_position:]
                        )

                # 替换 run_tests() 为 unittest.main(testRunner=CustomTextTestRunner())
                content = re.sub(
                    r"run\_tests\(\)",
                    "unittest.main(testRunner=CustomTextTestRunner())",
                    content,
                )

                # 将修改后的内容写回文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="original pytorch test file")
    args = parser.parse_args()
    modify_src_code(args.filename, "npu")
