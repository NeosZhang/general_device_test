import os
import re
import ditorch


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


def process_src_code(src: str):
    device_npu = ditorch.framework.split(":")[0]
    import_patch = """import torch
import ditorch
"""
    imports_text, functions_text = split_script(src)

    # 对某些device不支持的符号进行mock
    # torch.cuda.get_device_capability()
    mock_code = ""
    if device_npu == "torch_npu":
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

    # 通过环境变量，设置是否需要打印跳过测例的详细信息
    display_skipped_tests = os.environ.get("DISPLAY_SKIPPED_TESTS")
    if display_skipped_tests in ["True", "true", "1"]:
        import_unittest_module_code = """
import unittest
"""
        custom_test_runner_code = r"""
import os
import json
class CustomTextTestResult(unittest.TextTestResult):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_tests = []
        self.all_EF_infos = []

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.skipped_tests.append((test, reason))

    def addFailure(self, test, err):
        super().addFailure(test, err)

        self.all_EF_infos.append((test, err))

    def addError(self, test, err):
        super().addError(test, err)
        
        self.all_EF_infos.append((test, err))


    def printErrors(self):
        super().printErrors()

        if self.skipped_tests:
            self.stream.writeln(self.separator1)
            for test, reason in self.skipped_tests:
                self.stream.writeln(f"Skip {self.getDescription(test)}: {reason}")

        # 将异常信息转换为字符串
        # error_message = self._exc_info_to_string(err, test)
        current_file_path = os.path.abspath(__file__)
        desired_dir = "origin_torch"
        desired_path = current_file_path.split(desired_dir)[0] + desired_dir
        parent_directory = os.path.dirname(desired_path)
        file1 = parent_directory + "/unsupported_test_cases/test_failures_errors.json"
        # 检查文件是否存在
        if not os.path.exists(file1):
            # 如果文件不存在，创建一个空的 JSON 文件
            with open(file1, 'w') as f:
                json.dump({}, f)
        fr = open(file1)
        content = json.load(fr)
        for test, err in self.all_EF_infos:
            exctype, value, tb = err
            need_value = str(value).split("\n\nTo execute this test,")[0]
            content[str(test)] = [f"{type(value).__name__}", [f"{need_value}"]]
        with open(file1, mode="w") as fp:
            fp.write("{\n")
            length = len(content.keys()) - 1
            for i, (key, (value1, value2)) in enumerate(content.items()):
                if i < length:
                    fp.write(f"  \"{key}\": [\"{value1}\", [\"{value2}\"]]" + ",\n")
                else:
                    fp.write(f"  \"{key}\": [\"{value1}\", [\"{value2}\"]]" + "\n")
            fp.write("}\n")
        fr.close()

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
    process_src_code(args.filename)
