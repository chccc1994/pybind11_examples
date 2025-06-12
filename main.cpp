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
    fs::path script_path = fs::current_path() / "core_reid";
    std::string python_path = script_path.string() + ";D:\\Program\\Anaconda\\envs\\py39\\Lib";

    _putenv_s("PYTHONHOME", "D:\\Program\\Anaconda\\envs\\py39");
    _putenv_s("PYTHONPATH", python_path.c_str());
    py::scoped_interpreter guard{};

    try
    {
        // [1] 读取图像
        py::module cvmodule = py::module::import("search_fgx");
        cvmodule.attr("init")();

        // [2] 图像参数传入
        // a. 读取图像
        cv::Mat mat = cv::imread("123.jpg"); // 确保文件存在
        if (mat.empty())
        {
            std::cerr << "Failed to load image!" << std::endl;
            return -1;
        }

        // b.将图像转换为 py::array
        // py::array_t<uint8_t> img_array(
        //     {mat.rows, mat.cols, mat.channels()},
        //     mat.ptr());

        // 创建 numpy 数组的形状和步长
        std::vector<size_t> shape;
        std::vector<size_t> strides;

        if (mat.channels() == 1)
        {
            shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols)};
            strides = {static_cast<size_t>(mat.cols), 1};
        }
        else
        {
            shape = {static_cast<size_t>(mat.rows), static_cast<size_t>(mat.cols), static_cast<size_t>(mat.channels())};
            strides = {static_cast<size_t>(mat.cols * mat.channels()), static_cast<size_t>(mat.channels()), 1};
        }

        // 创建 numpy 数组（不拷贝数据）
        py::array_t<unsigned char> img_array = py::array_t<unsigned char>(
            shape,
            strides,
            mat.data);

        // c. 导入 Python 模块并调用函数
        auto start = std::chrono::high_resolution_clock::now();
        py::object result = cvmodule.attr("person_reid")(img_array);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Recognition result: " << result.cast<std::string>() << "\n";
        std::cout << "person_reid time: " << duration.count() << " ms" << std::endl;
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