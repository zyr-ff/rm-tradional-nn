cmake_minimum_required(VERSION 3.16)
project(rm_armor_vision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
#  目录约定（按识别方法分离，共用模块独立）
#    common/       共用：Armor 数据结构、相机参数、PnP 解算、
#                        视频主程序(main.cpp, 用 USE_NN 宏切换两种检测器)
#    traditional/  传统方法(D2)：颜色阈值找灯条 -> 灯条配对 -> MLP 数字识别
#    neural/       神经网络方法(D3)：YOLO v8n 直接输出四关键点+颜色+装甲类别
#    models/       模型文件：mlp.onnx(传统数字识别) / Infantry-v8n*.onnx(NN检测)
#                         / label.txt(两类共用的 9 类标签)
#    tools/        离线分析脚本(py)，不属于构建目标
# ============================================================

# ===================== OpenCV =====================
# MSYS2 UCRT64 安装: pacman -S mingw-w64-ucrt-x86_64-opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ===================== ONNX Runtime =====================
# 注意: onnxruntime 只有 UCRT64 版! MSYS2 环境请用 UCRT64 终端:
#   pacman -S mingw-w64-ucrt-x86_64-onnxruntime
# 若 CMake 找不到，可显式指定: cmake .. -DONNXRUNTIME_DIR=/ucrt64
# 说明: 该库为"可选"——即使还没装，仅 OpenCV 部分也能先编译跑通(见 main.cpp)。
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime/onnxruntime_cxx_api.h          # MSYS2 布局
          onnxruntime/core/session/onnxruntime_cxx_api.h  # 官方预编译包布局
    HINTS ${ONNXRUNTIME_DIR} /ucrt64/include /mingw64/include $ENV{ONNXRUNTIME_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_DIR} /ucrt64/lib /mingw64/lib $ENV{ONNXRUNTIME_DIR}/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
else()
    message(WARNING "ONNX Runtime 未找到 -> 神经网络/数字识别模块暂不可用。"
                    "可先只跑 OpenCV 基线，或设置 -DONNXRUNTIME_DIR=/mingw64")
endif()

# ===================== 源文件分组（显式列出，不用 GLOB，便于按方法分离）=====================
set(COMMON_INC ${CMAKE_SOURCE_DIR}/common/include)
set(TRAD_INC   ${CMAKE_SOURCE_DIR}/traditional/include)
set(NN_INC     ${CMAKE_SOURCE_DIR}/neural/include)

set(COMMON_SRC common/src/main.cpp
               common/src/pnp_solver.cpp)
set(TRAD_SRC   traditional/src/armor_detector.cpp
               traditional/src/number_classifier.cpp)
set(NN_SRC     neural/src/neural_detector.cpp)

# ===================== 主程序：视频流水线 =====================
# main.cpp 内用 #define USE_NN 切换 神经网络/传统 两种检测器，
# 因此本目标同时包含两套检测器源码（默认启用 USE_NN，见源码开关）。
add_executable(${PROJECT_NAME} ${COMMON_SRC} ${TRAD_SRC} ${NN_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_INC} ${TRAD_INC} ${NN_INC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARY})
endif()

# ===================== 传统方法 · 误差评估工具 (D4) =====================
# 复用传统检测/数字识别 + 共用 PnP，单独编 exe，不链接 main.cpp
add_executable(eval_gt
    traditional/tools/eval_gt.cpp
    ${TRAD_SRC}
    common/src/pnp_solver.cpp)
target_include_directories(eval_gt PRIVATE ${COMMON_INC} ${TRAD_INC})
target_link_libraries(eval_gt ${OpenCV_LIBS})
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(eval_gt ${ONNXRUNTIME_LIBRARY})
endif()

# ===================== 神经网络方法 · 诊断小程序 (D3) =====================
# 不依赖显示器：跑一次前向，把模型张量输出与解算结果打印到终端。
add_executable(test_nn
    neural/tools/test_nn.cpp
    ${NN_SRC})
target_include_directories(test_nn PRIVATE ${COMMON_INC} ${NN_INC})
target_link_libraries(test_nn ${OpenCV_LIBS})
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(test_nn ${ONNXRUNTIME_LIBRARY})
endif()

# ===================== 神经网络方法 · 误差评估工具 (D4) =====================
# 与 eval_gt 完全一致的 真值/匹配/指标，只把检测器换成 ArmorDetectorNN。
add_executable(eval_gt_nn
    neural/tools/eval_gt_nn.cpp
    ${NN_SRC}
    common/src/pnp_solver.cpp)
target_include_directories(eval_gt_nn PRIVATE ${COMMON_INC} ${NN_INC})
target_link_libraries(eval_gt_nn ${OpenCV_LIBS})
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(eval_gt_nn ${ONNXRUNTIME_LIBRARY})
endif()

# ===================== 运行时 DLL 自动拷贝 =====================
# Windows 加载 DLL 的搜索顺序: ① exe 所在目录 ② System32 ③ Windows 目录 ④ ... ⑤ PATH。
# ⚠️ 某些软件会往 C:\Windows\System32 装旧版 onnxruntime.dll(实测有 1.17.1),
#    它排在 PATH 之前, 会"截胡" MSYS2 的正确版本, 导致运行时报:
#    "The requested API version [29] is not available ... Current ORT Version is: 1.17.1"
# 解决: 构建后自动把与头文件匹配的 onnxruntime DLL 拷到各 exe 旁边(①优先级最高)。
if(ONNXRUNTIME_LIBRARY)
    get_filename_component(ORT_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
    set(ORT_DLLS)
    foreach(d ${ORT_LIB_DIR}/../bin/onnxruntime.dll
              ${ORT_LIB_DIR}/../bin/onnxruntime_providers_shared.dll)
        if(EXISTS ${d})
            list(APPEND ORT_DLLS ${d})
        endif()
    endforeach()
    if(ORT_DLLS)
        foreach(tgt rm_armor_vision eval_gt test_nn eval_gt_nn)
            add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${ORT_DLLS} $<TARGET_FILE_DIR:${tgt}>
                COMMENT "拷贝 onnxruntime DLL 到 ${tgt} 所在目录(避开 System32 旧版)")
        endforeach()
    endif()
endif()

# 在 MSYS2 UCRT64 终端运行 exe 时 DLL 已由上面的 POST_BUILD 步骤放到 exe 旁, 无需手动拷贝。
# (不要依赖 PATH: System32 里的旧版 onnxruntime.dll 优先级高于 PATH。)
