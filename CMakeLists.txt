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
# 添加pybind11子模块
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)

# ========================== YAML-CPP ==========================
set(YAML_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/YAML_CPP)

include_directories(${YAML_CPP_ROOT}
                    ${YAML_CPP_ROOT}/include)

set(YAML_LIB_ROOT ${YAML_CPP_ROOT}/lib)
link_directories(${YAML_LIB_ROOT})
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(YAML_LIB ${YAML_LIB_ROOT}/yaml-cppd.lib)
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  set(YAML_LIB ${YAML_LIB_ROOT}/yaml-cpp.lib)
endif()

# ========================== JSON CPP ==========================
set(JSONCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/jsoncpp)
include_directories(${JSONCPP_DIR})
include_directories(${JSONCPP_DIR}/include)
aux_source_directory(${JSONCPP_DIR}/src JSON_SRC_LIST)
# 添加库文件路径
link_directories(${JSONCPP_DIR}/lib)

# ========================== Eigen3 ==========================
set(Eigen3_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen3)
set(Eigen3_INCLUDE_DIRS ${Eigen3_DIRS}/include/eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

# ========================== OpenCV ==========================
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv4.10)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 OpenCV 的 DLL 和 LIB 文件路径
set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
set(OpenCV_LIB_DIR "${OpenCV_DIR}/x64/vc16/lib")

# 根据 CMAKE_BUILD_TYPE 复制文件
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    # 复制 Debug 模式的 DLL 和 LIB 文件
    file(GLOB OpenCV_DLL_DIRS "${OpenCV_BIN_DIR}/*d.dll")
    file(GLOB OpenCV_LIB_DIRS "${OpenCV_LIB_DIR}/*d.lib")
    file(COPY ${OpenCV_BIN_DIRS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${OpenCV_LIB_DIRS} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    # 复制 Release 模式的 DLL 和 LIB 文件
    file(GLOB OpenCV_DLL_DIRS "${OpenCV_BIN_DIR}/*.dll")
    file(GLOB OpenCV_LIB_DIRS "${OpenCV_LIB_DIR}/*.lib")
    file(COPY ${OpenCV_DLL_DIRS} DESTINATION ${CMAKE_BINARY_DIR}/Release)
    file(COPY ${OpenCV_LIB_DIRS} DESTINATION ${CMAKE_BINARY_DIR}/Release)
endif()

# ========================== CUDA ==========================
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CMAKE_CUDA_ARCHITECTURES 86)  # 例如，对于 Ampere 架构的 GPU
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
enable_language(CUDA)


# --------------------------- 源代码 ---------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# 源文件和头文件
set(SOURCES
  main.cpp
  ${JSON_SRC_LIST}
)

# 添加可执行文件
add_executable(${PROJECT_NAME} 
    ${SOURCES}
)

# 设置脚本路径
set(SCRIPTS_SRC_DIR "${CMAKE_SOURCE_DIR}/scripts")
set(SCRIPTS_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/scripts")

# 获取所有 .py 文件
file(GLOB SCRIPT_FILES "${SCRIPTS_SRC_DIR}/*.py")

# 创建一个 custom target 用于复制 .py 文件
add_custom_target(copy_scripts ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SCRIPTS_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SCRIPT_FILES} "${SCRIPTS_DST_DIR}"
    COMMENT "Copying Python scripts to output directory"
)

# 链接库
if(WIN32)
  target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${YAML_LIB}
    pybind11::embed
    jsoncpp
  )
  if (USE_CUDA)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
  endif ()
endif()