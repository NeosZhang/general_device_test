import os
import re
import re


def split_script(input_file):
    # 用于匹配 import 和 from ... import 语句（包括多行）
    import_pattern = re.compile(r"^\s*(import\s+\w+|from\s+\w+(\.\w+)*\s+import\s+)")
    continuation_pattern = re.compile(r"^\s*[\(\,]")

    # 读取原始脚本
    with open(input_file, "r") as f:
        lines = f.readlines()

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
        "onlyCUDAAndPRIVATEUSE1": "onlyCUDAAndPRIVATEUSE1",
        "TEST_MULTIGPU": "TEST_MULTIGPU",
        "TEST_CUDA_GRAPH": "TEST_CUDA_GRAPH",
        "TEST_CUDA": "TEST_CUDA",
        "TEST_CUDNN": "TEST_CUDNN",
        "torch.cuda": "torch.cuda",
        "onlyCUDA": "onlyCUDA",
        "RUN_CUDA": "RUN_CUDA",
        "dtypesIfCUDA": "dtypesIfCUDA",
        ".cuda(": ".cuda(",
        '"cuda"': '"cuda"',
        "'cuda'": "'cuda'",
        "cuda:": "cuda:",
    }
    if device_code == "cuda":
        pass
    elif device_code == "npu":
        import_torch = "import torch_npu\nimport torch_npu.testing\nfrom torch.testing._internal.common_utils import TEST_PRIVATEUSE1\n\
from torch.testing._internal.common_device_type import onlyPRIVATEUSE1, dtypesIfPRIVATEUSE1\n\
TEST_MULTINPU = TEST_PRIVATEUSE1 and torch_npu.npu.device_count() >= 2\n\
RUN_PRIVATEUSE1_MULTI_GPU = TEST_MULTINPU\n\
TEST_PRIVATEUSE1_VERSION = 5130\n\
RUN_PRIVATEUSE1 = True\n\
TEST_NPU = torch_npu.npu.is_available()\n\
TEST_BFLOAT16 = TEST_NPU and torch_npu.npu.is_bf16_supported()\n\
RUN_PRIVATEUSE1_HALF = RUN_PRIVATEUSE1\n"

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
    else:
        raise ValueError("invald device code! The legal device code are: cuda, npu.")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="original pytorch test file")
    args = parser.parse_args()
    modify_src_code(args.filename, "npu")
