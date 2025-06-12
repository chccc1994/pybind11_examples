以下是 C++ 调用 Python 方法的对比表格，涵盖 PyBind11、Python C API 和 系统调用 三种主要方式：

| 特性       | PyBind11                    | Python C API                   | 系统调用 (QProcess等)      |
| ---------- | --------------------------- | ------------------------------ | -------------------------- |
| 易用性     | ⭐⭐⭐⭐⭐ (类C++语法)           | ⭐⭐ (需手动管理引用/异常)       | ⭐⭐⭐ (简单命令调用)         |
| 性能       | ⭐⭐⭐⭐ (直接内存共享)         | ⭐⭐⭐⭐⭐ (最底层无额外开销)       | ⭐ (进程间通信开销大)       |
| 数据交互   | ⭐⭐⭐⭐⭐ (自动类型转换)        | ⭐⭐⭐ (需手动转换)               | ⭐ (仅文本/二进制流)        |
| 线程安全   | ⭐⭐⭐ (需手动管理GIL)         | ⭐⭐ (完全手动管理GIL)           | ⭐⭐⭐⭐ (独立进程)            |
| 依赖管理   | ⭐⭐⭐ (需Python环境)          | ⭐⭐⭐ (需Python开发头文件)       | ⭐ (需完整Python环境)       |
| 适用场景   | 高性能嵌入/复杂数据交互     | 底层控制/轻量级嵌入            | 简单脚本调用/黑盒执行      |
| 代码示例   | py::module::import("numpy") | PyImport_ImportModule("numpy") | system("python script.py") |
| 跨平台支持 | ⭐⭐⭐⭐⭐                       | ⭐⭐⭐⭐⭐                          | ⭐⭐⭐ (路径/Shell差异)       |
| 调试难度   | ⭐⭐ (C++异常+Python错误混合) | ⭐⭐⭐ (纯C错误处理)              | ⭐⭐⭐⭐ (独立日志)            |
| 维护成本   | ⭐⭐ (需同步C++/Python接口)   | ⭐⭐⭐ (接口变更需手动调整)       | ⭐⭐⭐ (需维护脚本路径)       |


# pybind11教程
[toc]
## 1. pybind11简介
[pybind11](https://github.com/pybind/pybind11.git) 是一个轻量级的头文件库，用于在 Python 和 C++ 之间进行互操作。它允许 C++ 代码被 Python 调用，反之亦然。

pybind11 的优点包括：

- 易于使用：pybind11 的 API 简单易懂，即使是初学者也可以快速上手。
- 高性能：pybind11 使用 C++ 的编译器来生成 Python 的 C 扩展，因此性能非常高。
- 跨平台：pybind11 支持 Windows、Linux 和 macOS。

pybind11 的使用方法非常简单。只需在 C++ 代码中包含 pybind11 头文件，然后使用 pybind11 提供的 API 来将 C++ 类型和函数暴露给 Python。

## 2. cmake使用pybind11教程

这部分代码开源在：

```cmake
cmake_minimum_required(VERSION 3.10)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(SOFT_VERSION V${VERSION_MAJOR}.${VERSION_MINOR})

# 定义项目名称变量
set(PROJECT_NAME "pybind11_examples")
project(${PROJECT_NAME} )

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/utf-8)                       # MSVC启用UTF-8编码
endif()

# ---------------------- 项目配置 ----------------------

# -------------- Python  ------------------#
# 添加子模块
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)

# --------------------------- 源代码 ---------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# 源文件和头文件
set(SOURCES
  main.cpp
)

# 添加可执行文件
add_executable(${PROJECT_NAME} 
    ${SOURCES}
)

# 复制脚本到可执行目录下
set(SCRIPTS_SRC_DIR "${CMAKE_SOURCE_DIR}/scripts")
set(SCRIPTS_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/scripts")

file(GLOB SCRIPT_FILES "${SCRIPTS_SRC_DIR}/*.py")

add_custom_target(copy_scripts ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCRIPTS_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SCRIPT_FILES} "${SCRIPTS_DST_DIR}"
    COMMENT "Copying Python scripts to output directory"
)

# 链接库
if(WIN32)
  target_link_libraries(${PROJECT_NAME}
    pybind11::embed
  )
endif()
```


- 实现一个 `C++` 代码，然后通过 `pybind11` 包装给 `python`
```c++
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <windows.h>
#include <cstdlib> // for std::setenv
#include <iostream>
#include <filesystem>
#include <json/json.h>
#include <opencv2/opencv.hpp>
namespace py = pybind11;
namespace fs = std::filesystem;
int main()
{
    fs::path script_path = fs::current_path() / "scripts";
    std::string python_path = script_path.string() + ";D:\\Program\\Anaconda\\envs\\py39\\Lib";

    _putenv_s("PYTHONHOME", "D:\\Program\\Anaconda\\envs\\py39");
    _putenv_s("PYTHONPATH", python_path.c_str());
    py::scoped_interpreter guard{};
    try
    {
        // 1. numpy导入验证
        auto np = py::module::import("numpy");
        std::cout << "numpy path: " << np.attr("__file__").cast<std::string>() << std::endl;

        // 2. numpy模块测试
        py::module npmodule = py::module::import("npmodule");

        std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
        py::object result = npmodule.attr("compute_mean")(data);

        double mean = result.cast<double>();
        std::cout << "Mean: " << mean << std::endl;
    }
    catch (py::error_already_set &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        // 10. 错误恢复机制
        py::exec("import sys; print(sys.path)");
        return -1;
    }

    py::finalize_interpreter();
    return 0;
}
```


```python
import numpy as np

def compute_mean(arr):
    arr = np.array(arr)
    return np.mean(arr)

```

---
## 3. pybind11
pybind11 逐渐支持了越来越多的 C++ 特性，包括：

- 类和对象
- 模板
- 继承
- 多态
- 异常处理
- 线程安全
- 动态类型

pybind11 的开发工作一直在进行中，Wenzel Jakob 和其他开发人员不断添加新的特性和功能。


pybind11 是一个非常强大的工具，可以用于各种任务。它可以用于将 C++ 代码与 Python 脚本集成，也可以用于创建 Python 的 C++ 扩展。**已经成为 C++ 和 Python 互操作领域的事实标准。**

## 4. pyarmor 库

| 特性       | 技术实现               | 应用场景          |
| ---------- | ---------------------- | ----------------- |
| 多层混淆   | 控制流扁平化+指令替换  | 防止逆向工程      |
| 量子加密   | CRYSTALS-Kyber算法集成 | 金融级安全需求    |
| 硬件绑定   | TPM 2.0芯片支持        | 防止非法设备运行  |
| 跨平台编译 | 生成ARM/x86/WASM字节码 | 边缘计算场景      |
| 动态许可证 | 区块链智能合约验证     | 软件订阅制管理    |
| 反调试注入 | 内存校验+断点检测      | 对抗IDA Pro等工具 |


[Pyarmor](https://github.com/dashingsoft/pyarmor) 是一个用于加密和保护 Python 脚本的工具。它能够在运行时刻保护 Python 脚本代码不被泄露，设置加密后脚本的使用期限，绑定加密脚本到硬盘、网卡等硬件设备。[Pyarmor 9.0 用户文档。](https://pyarmor.readthedocs.io/zh/latest/)
功能特点:

+ 无缝替换: 加密后的脚本依然是一个有效的 .py 文件，在大多数情况下可以直接替换原来的 .py 脚本，而不影响脚本的使用。

+ 均衡加密: 提供了丰富的加密选项来平衡安全性和性能，能够满足大多数应用对安全性和性能的要求。

+ 不可逆加密: 能够直接重命名源代码中的函数，类，方法，变量和参数。

+ 转换成为 C 代码: 能够把模块中部分函数转换成为 C 代码，然后使用高优化选项直接编译 C 代码为机器指令来保护 Python 函数

+ 限制加密脚本的使用范围: 可以绑定加密脚本到指定的设备或者设置加密脚本的有效期

+ Themida 保护: 使用 Themida 保护加密脚本（仅 Windows 平台可用）

```python
# 简单加密
pyarmor gen foo.py
pyarmor g foo.py
pyarmor generate foo.py

# 这个命令会生成一个加密脚本 dist/foo.py ，这也是一个正常的 Python 脚本，可以直接使用 Python 解释器执行:

python dist/foo.py
# 查看所有生成的文件:
#ls dist/
#...    foo.py
#...    pyarmor_runtime_000000
#除了加密脚本之外，可以看到还有另外一个目录 pyarmor_runtime_000000 ，这是运行加密脚本所依赖的一个 Python 包 。
```