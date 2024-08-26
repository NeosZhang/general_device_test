import os
import re


def display_skipped_tests():
    import_unittest_module_code = """
    import unittest
    """

    custom_test_runner_code = """
    from unittest.runner import TextTestResult
    # Custom test result class to print skipped tests and their reasons
    class CustomTextTestResult(unittest.TextTestResult):
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            print(f"Skipped {{test}}: {{reason}}")

    class CustomTextTestRunner(unittest.TextTestRunner):
        resultclass = CustomTextTestResult

    """

    # 定义你要搜索测试文件的目录
    test_directory = "./modified_tests"  # 你可以修改为你的实际测试文件目录

    # 遍历目录，找到所有以test_开头的.py文件
    for root, dirs, files in os.walk(test_directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "unittest" not in content:
                    content = import_unittest_module_code + content
                # 检查是否已经包含CustomTextTestRunner代码，避免重复添加
                if "CustomTextTestRunner" not in content:
                    match = re.search(r"(import unittest\s*\n)", content)
                    if match:
                        # 在 import unittest 后插入自定义测试运行器代码
                        insert_position = match.end()
                        content = (
                            content[:insert_position]
                            + custom_test_runner_code
                            + content[insert_position:]
                        )

                # 替换 unittest.main() 为 unittest.main(testRunner=CustomTextTestRunner())
                content = re.sub(
                    r"run\_tests\(\)",
                    "unittest.main(testRunner=CustomTextTestRunner())",
                    content,
                )

                # 将修改后的内容写回文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
