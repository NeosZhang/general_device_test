import os
import re


def split_script(input_file):
    """用于将测试脚本test_xxx.py拆分为import部分和代码主体部分"""
    with open(input_file, "r") as f:
        lines = f.readlines()
    import_lines = []
    for line in lines:
        if line.startswith("#"):
            import_lines.append(line)
            continue
        elif "=" in line or ":" in line:
            break
        else:
            import_lines.append(line)
    while import_lines[-1].startswith("#"):
        import_lines.pop()

    imports_text = "".join(import_lines)
    functions_text = "".join(lines[len(import_lines) :])
    return imports_text, functions_text


def device_patch_for_test(device_code: str):
    import_patch = ""
    symbol_replace_map = {}
    if device_code == "cuda":
        pass
    elif device_code == "npu":
        import_patch = """import torch_npu
import torch_npu.testing
from torch.testing._internal.common_utils import TEST_PRIVATEUSE1
from torch.testing._internal.common_device_type import onlyPRIVATEUSE1, dtypesIfPRIVATEUSE1
TEST_MULTINPU = TEST_PRIVATEUSE1 and torch_npu.npu.device_count() >= 2
RUN_PRIVATEUSE1_MULTI_GPU = TEST_MULTINPU
TEST_PRIVATEUSE1_VERSION = 5130
RUN_PRIVATEUSE1 = True
TEST_NPU = torch_npu.npu.is_available()
TEST_BFLOAT16 = TEST_NPU and torch_npu.npu.is_bf16_supported()
RUN_PRIVATEUSE1_HALF = RUN_PRIVATEUSE1
"""

        symbol_replace_map["onlyCUDAAndPRIVATEUSE1"] = "onlyPRIVATEUSE1"
        symbol_replace_map["TEST_MULTIGPU"] = "TEST_MULTINPU"
        symbol_replace_map["TEST_CUDA_GRAPH"] = "TEST_PRIVATEUSE1"
        symbol_replace_map["TEST_CUDA"] = "TEST_PRIVATEUSE1"
        symbol_replace_map["TEST_CUDNN"] = "TEST_PRIVATEUSE1"
        symbol_replace_map["torch.cuda"] = "torch_npu.npu"
        symbol_replace_map["onlyCUDA"] = "onlyPRIVATEUSE1"
        symbol_replace_map["RUN_CUDA"] = "RUN_PRIVATEUSE1"
        symbol_replace_map["dtypesIfCUDA"] = "dtypesIfPRIVATEUSE1"
        symbol_replace_map[".cuda("] = ".npu("
        symbol_replace_map['"cuda"'] = '"npu"'
        symbol_replace_map["'cuda'"] = "'npu'"
        symbol_replace_map["cuda:"] = "npu:"
    elif device_code == "muxi":
        pass
    elif device_code == "ditorch":
        import_patch = """import torch
import ditorch
"""
    else:
        raise ValueError(
            "invald device code! The legal device code are: cuda, npu, muxi."
        )
    return import_patch, symbol_replace_map


def replace_in_text(text, mapping):
    # 将映射关系字典的键转换为正则表达式可以接受的格式，确保能正确处理特殊字符
    pattern = re.compile("|".join(re.escape(key) for key in mapping.keys()))
    # 使用正则表达式替换文本中的键
    return pattern.sub(lambda x: mapping[x.group()], text)


def modify_src_code(src: str, device_code: str):
    import_patch, symbol_replace_map = device_patch_for_test(device_code)
    imports_text, functions_text = split_script(src)

    # 对某些device不支持的符号进行mock
    # torch.cuda.get_device_capability()
    mock_code = """
from unittest.mock import patch
patch('torch.cuda.get_device_capability', return_value=(8, 0)).start()

import torch_npu
if not hasattr(torch._C, '_cuda_setStream'):
    def _cuda_setStream(*args, **kwargs):
        pass
    setattr(torch._C, '_cuda_setStream', _cuda_setStream)
patch('torch._C._cuda_setStream', new=torch_npu._C._npu_setStream).start()

if not hasattr(torch._C, '_cuda_setDevice'):
    def _cuda_setDevice(*args, **kwargs):
        pass
    setattr(torch._C, '_cuda_setDevice', _cuda_setDevice)
patch('torch._C._cuda_setDevice', new=torch_npu._C._npu_setDevice).start()
"""
    imports_text = import_patch + mock_code + imports_text

    # 添加import_patch，并根据symbol_replace_map进行符号替换
    if symbol_replace_map:
        functions_text = replace_in_text(functions_text, symbol_replace_map)

    # 通过环境变量，设置是否需要打印跳过测例的详细信息
    display_skipped_tests = os.environ.get("DISPLAY_SKIPPED_TESTS")
    if display_skipped_tests in ["True", "true", "1"]:
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
        if "import unittest\n" not in imports_text:
            imports_text = import_unittest_module_code + imports_text
            # 检查是否已经包含CustomTextTestRunner代码，避免重复添加
        if "CustomTextTestRunner" not in functions_text:
            functions_text = custom_test_runner_code + functions_text

        functions_text = re.sub(
            r"run\_tests\(\)",
            "unittest.main(testRunner=CustomTextTestRunner())",
            functions_text,
        )

    with open(src, "w", encoding="utf-8") as file:
        file.write(imports_text + functions_text)


if __name__ == "__main__":
    import argparse

    os.environ["DISPLAY_SKIPPED_TESTS"] = "True"
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="original pytorch test file")
    args = parser.parse_args()
    modify_src_code(args.filename, "ditorch")
