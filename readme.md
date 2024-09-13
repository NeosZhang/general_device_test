# 1. 背景
1. 本工具是ditorch的子仓库，用于在不同的国产设备上运行PyTorch官方的测试用例
2. 目前只适配了torch_npu

# 2. 适配逻辑
由于PyTorch官方的测试用例主要是针对CUDA设备的，所以需要一些改动才能适配国产设备。 \
通过运行如下命令，即可完成对PyTorch测试脚本的处理
```
python main.py
```
main.py主要进行以下工作：
    1. 克隆PyTorch官方仓库(以下称为origin_torch) \
    2. 过滤部分origin_torch中不必要的测试脚本，并将剩余的测试脚本拷贝到processed_tests路径下 \
    3. 对processed_tests路径中的测试脚本进行处理

# 3. 测试方法
使用的测试框架是unittest，支持以下的命令行选项：
1. 执行单个测试脚本 \
以test_nn.py为例 
```
python test_nn.py
```
2. 测试单个test_method \
可以通过以下命令来运行某个特定的test_method
```
python -m unittest test_file.Test_Class.test_method
```

3. -k \
只运行匹配模式或子字符串的测试方法和类。 \
以test_autocast.py为例：
```
python test_autocast.py -v -k test_autocast_nn_fp32
```
4. -f --failfast
当出现第一个错误或者失败时，停止运行测试。

5. 跳过测试用例 
```
export DISABLED_TESTS_FILE=./unsupported_test_cases/.pytorch-disabled-tests.json

```
如果不是在test目录下运行测试用例，需要传入.pytorch-disabled-tests.json的绝对路径。

4. 错误记录 \
测试结果为Error和Failure的测试会被自动记录到unsupported_test_cases下的json文件 \
记录格式为：\
`"{test_name}": ["{Error type}", ["{error reason}"]]` \
可以手动对json文件进行编辑。
