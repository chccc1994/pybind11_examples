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

        // [3] 读取图像
        py::module cvmodule = py::module::import("cvmodule");
        // cvmodule.attr("show_image_basic")();

        // [4] 图像参数传入
        // a. 读取图像
        cv::Mat image = cv::imread("1.jpg"); // 确保文件存在
        if (image.empty())
        {
            std::cerr << "Failed to load image!" << std::endl;
            return -1;
        }

        // b.将图像转换为 py::array
        py::array_t<uint8_t> img_array(
            {image.rows, image.cols, image.channels()},
            image.ptr());
        // c. 导入 Python 模块并调用函数
        result = cvmodule.attr("get_image_shape")(img_array);

        // d. 打印返回值
        auto shape = result.cast<py::tuple>();
        std::cout << "Returned shape: ("
                  << shape[0].cast<int>() << ", "
                  << shape[1].cast<int>() << ", "
                  << shape[2].cast<int>() << ")" << std::endl;

        // [5] torch测试
        py::dict info = cvmodule.attr("get_torch_info")().cast<py::dict>();
        std::cout << "Torch version: " << info["torch_version"].cast<std::string>() << "\n";
        std::cout << "CUDA available: " << info["cuda_available"].cast<bool>() << "\n";
        std::cout << "CUDA version: " << info["cuda_version"].cast<std::string>() << "\n";
        std::cout << "cuDNN version: " << info["cudnn_version"].cast<int>() << "\n";
        std::cout << "Device count: " << info["device_count"].cast<int>() << "\n";

        auto devices = info["devices"].cast<std::vector<py::dict>>();
        for (const auto &device : devices)
        {
            std::cout << "  GPU " << device["index"].cast<int>()
                      << ": " << device["name"].cast<std::string>()
                      << " (" << device["total_memory_MB"].cast<double>() << " MB)"
                      << ", Compute capability: " << device["compute_capability"].cast<std::string>()
                      << "\n";
        }

        // [6] json测试
        py::module_ jsonmodule = py::module_::import("jsonmodule");
        result = jsonmodule.attr("get_data")(); // 调用函数

        // 将 py::object 转换为 JSON 字符串
        py::module_ json_module = py::module_::import("json");
        std::string json_str = json_module.attr("dumps")(result).cast<std::string>();

        // 使用 jsoncpp 解析字符串
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errs;
        std::istringstream ss(json_str);
        if (Json::parseFromStream(builder, ss, &root, &errs))
        {
            std::cout << "Name: " << root["name"].asString() << std::endl;
            std::cout << "Age: " << root["age"].asInt() << std::endl;
            std::cout << "Languages: ";
            for (const auto &lang : root["languages"])
            {
                std::cout << lang.asString() << " ";
            }
            std::cout << std::endl;
        }
        else
        {
            std::cerr << "Failed to parse JSON: " << errs << std::endl;
        }
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