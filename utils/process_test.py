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
    device_torch = ditorch.framework.split(":")[0]
    import_patch = """import torch
import ditorch
"""
    imports_text, functions_text = split_script(src)

    # 对某些device不支持的符号进行mock
    # torch.cuda.get_device_capability()
    mock_code = ""
    if device_torch == "torch_npu":
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
    custom_test_code = r"""import os
import json
import unittest

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
        disabled_test_json = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/unsupported_test_cases/test_failures_errors.json"
        if not os.path.exists(disabled_test_json) or os.path.getsize(disabled_test_json) == 0:
            with open(disabled_test_json, 'w') as f:
                json.dump({}, f)
            f.close()
        with open(disabled_test_json, 'a+') as f:
            try:
                # 如果文件为空，则初始化为空字典
                content = json.load(f) if f.read() else {}
                f.seek(0)  # 将文件指针移到开头
                f.truncate()  # 清空文件内容
            except json.JSONDecodeError:
                # 如果 JSON 解析失败，初始化为空字典
                content = {}
                f.seek(0)  # 将文件指针移到开头
                f.truncate()  # 清空文件内容
            for test, err in self.all_EF_infos:
                exctype, value, tb = err
                need_value = str(value).split("\n")[0]
                if str(test) not in content:
                    content[str(test)] = [f"{exctype.__name__}", [f"{need_value}"]]
            print("content = ", content)
            json.dump(content, f, indent=4)

class CustomTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTextTestResult(self.stream, self.descriptions, self.verbosity)
"""
    imports_text = import_patch + mock_code + imports_text + custom_test_code

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
