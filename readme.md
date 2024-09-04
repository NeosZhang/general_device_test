# 1. 背景
1. 本工具是ditorch的子仓库，用于在不同的国产设备上运行torch官方的测试。由于torch官方的测试主要是针对CUDA设备的，所以需要一些改动才能适配国产设备。
2. 目前该工具暂时只完成了对NPU设备的适配。

# 2. 适配逻辑
```
python main.py
```
1. 克隆PyTorch官方仓库(以下称为origin_torch)和设备对应版本的torch(以下称为device_torch)
3. 过滤部分origin_torch中不必要的测试脚本，并将剩余的测试脚本拷贝到processed_tests路径下
4. 对processed_tests路径中的测试脚本进行处理
5. processed_tests中的测试脚本即为device所有的测试脚本

# 3. 测试方法
1. 执行单个测试脚本，以test_autocast.py为例 \
```
python test_autocast.py
```
2. 执行具体用例 \
通过-k参数传入具体的用例名。以test_autocast.py为例：\
```
python test_autocast.py -v -k test_autocast_nn_fp32
```

3. 跳过测试用例 \
```
export DISABLED_TESTS_FILE=./unsupported_test_cases/.pytorch-disabled-tests.json

```
如果不是在test目录下运行测试用例，需要传入.pytorch-disabled-tests.json的绝对路径。

4. 显示跳过的测试用例 \
在main.py运行前设置`DISPLAY_SKIPPED_TESTS`环境变量
```
export DISPLAY_SKIPPED_TESTS=True
```
<center>
    <img src='skipped_display.png' alt="display skipped testes">
</center>

5. 抓取失败日志到json文件，避免下次测试 \
首先，测试时抓取测试log：
```bash
python test_xxx.py 2>&1 | tee test_xxx.log
```
然后运行抓取log脚本：
```bash
python get_EF_test_from_log.py --file test_xxx.log
```
