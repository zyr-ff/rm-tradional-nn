# 视觉第二周作业 · C++ 工程骨架

工具链：**MSYS2 + UCRT64 (MinGW-w64)**（无 Visual Studio 时的标准方案，OpenCV 与 ONNX Runtime 同工具链，免编译）。
> ⚠️ 必须用 **UCRT64** 环境：MSYS2 的 onnxruntime 包只提供 UCRT64/CLANG64 版，**MINGW64 里装不到**。

## 一、安装环境（最长的一条杆，立刻开始）

1. 下载安装 MSYS2：<https://www.msys2.org>，全程下一步。
2. 打开开始菜单里的 **MSYS2 UCRT64** 终端（⚠️ 不是 MSYS、不是 MINGW64），依次执行：
   ```bash
   pacman -Syu            # 更新包数据库，可能提示关闭终端，重开后再跑一次
   pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake \
             mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-onnxruntime
   ```
   （`Enter a selection` 直接回车装全部；`Proceed?` 按 `Y`）
3. 验证：在 UCRT64 终端输入 `g++ --version`（应显示 13.x/14.x）、`cmake --version`，
   以及 `pkg-config --exists opencv4 && echo "opencv OK"`。

## 二、下载模型素材（放到 `code/models/`）

- **装甲板识别网络 (NN方案)**：<https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models>
  在 `所有模型/onnx/` 目录里找 `Infantry-v8n-20260726.onnx` 下载。
  输出是 4 个角点+类别+颜色，关键点顺序 `0左上 1左下 2右下 3右上`（见 `camera_config.h`）。
- **数字识别网络 (两种方案共用)**：<https://github.com/chenjunnn/rm_auto_aim/tree/main/armor_detector/model>
  下载 `mlp.onnx` 与 `label.txt`（经典 MLP，输入 20×28 二值图）。
- 该仓库的 `armor_detector`（Detector/NumberClassifier/PnPSolver）就是**传统方案的完整范本**，D2 直接读它、移植它。

## 三、构建并跑通基线

在 **UCRT64 终端**里：
```bash
cd code
mkdir -p build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build . -j
./rm_armor_vision.exe
```
预期：弹出视频预览窗口，按 `q` 退出，并先打印相机内参/畸变系数。
能跑通就说明 OpenCV 链路 OK，**之后才写算法**。

> 暂时没装 ONNX Runtime 也能编译（CMake 里设为可选），只是 NN/数字模块不可用。
> ⚠️ **onnxruntime.dll 的坑**：Windows 加载 DLL 时 `System32` 优先于 `PATH`，
> 若系统里被别的软件装了旧版 onnxruntime（实测有 1.17.1），运行时报
> `The requested API version [29] is not available`。
> 本工程已在 CMake 里加 POST_BUILD 步骤：**每次构建自动把匹配版本的 DLL 拷到 exe 旁边**
> （exe 目录优先级最高），无需任何手动操作。

## 四、⚠️ 今天必须向老师确认的两件事（否则任务 2、4 做不了）

1. **带坐标真值的图片**在哪？（docx 里只给了视频+链接，没给真值图）
2. **装甲板真实尺寸**（小装甲/大装甲的宽高，单位 mm）。先填进 `camera_config.h` 的
   `ARMOR_SMALL_*` / `ARMOR_LARGE_*` 占位常量。

## 五、五天冲刺路线（详见对话）

- D1 环境+素材+确认真值/尺寸+跑通本基线
- D2 传统方案（移植 chenjunnn：找灯条→配对→MLP数字→solvePnP）
- D3 NN 方案（v8n onnx + ONNX Runtime 推理→四关键点→复用 MLP+PnP）
- D4 标注视频 + 真值图误差 Δx/Δy/Δz 与三维误差对比表
- D5 模块化(头/实现分离)+CMake+Git(仓库/PR/commit hash)+写 Markdown 报告

## 六、项目结构（按识别方法分离）

```
code/
├── CMakeLists.txt            # 4 个构建目标：rm_armor_vision / eval_gt / test_nn / eval_gt_nn
├── camera_sim.yaml           # 仿真相机内参（camera_config.h 的数据来源）
├── common/                   【共用模块】
│   ├── include/
│   │   ├── armor_types.h     #   Armor 数据结构（两套检测器的统一输出）
│   │   ├── camera_config.h   #   相机内参/畸变/装甲尺寸/关键点顺序
│   │   └── pnp_solver.h      #   PnP 位姿解算（IPPE）
│   └── src/
│       ├── main.cpp          #   视频主程序（#define USE_NN 切换两种检测器）
│       └── pnp_solver.cpp
├── traditional/              【传统方法 · D2】
│   ├── include/
│   │   ├── armor_detector.h      # 颜色阈值找灯条 → 灯条配对
│   │   └── number_classifier.h   # MLP 数字识别
│   ├── src/
│   │   ├── armor_detector.cpp
│   │   └── number_classifier.cpp
│   └── tools/
│       └── eval_gt.cpp           # 传统方法误差评估（任务4）
├── neural/                   【神经网络方法 · D3】
│   ├── include/
│   │   └── neural_detector.h     # v8n 推理：四关键点+颜色+装甲类别
│   ├── src/
│   │   └── neural_detector.cpp
│   └── tools/
│       ├── test_nn.cpp           # NN 检测端到端验证
│       └── eval_gt_nn.cpp        # NN 方法误差评估（任务4）
├── models/
│   ├── mlp.onnx                  # 传统方案：数字识别
│   ├── Infantry-v8n-*.onnx       # NN 方案：装甲板检测（fp16 原始 + fp32 转换版）
│   └── label.txt                 # 两类共用的 9 类标签
└── tools/                    # 离线模型分析脚本（py，调试/教学用，不参与构建）
```

**构建目标一览**：

| 目标 | 方法 | 用途 |
|---|---|---|
| `rm_armor_vision` | 两种（`USE_NN` 宏切换） | 视频流水线：检测→数字识别→PnP→画框导出 |
| `eval_gt` | 传统 | 真值图误差评估（Δx/Δy/Δz） |
| `test_nn` | 神经网络 | NN 检测端到端验证（打印角点/位姿） |
| `eval_gt_nn` | 神经网络 | 真值图误差评估（与传统同口径） |

## 七、构建（按方法分别运行）

```bash
cd code/build
cmake .. -DONNXRUNTIME_DIR=/ucrt64     # 首次或 CMakeLists 变更后
cmake --build . -j

# 传统方法：视频（把 main.cpp 的 #define USE_NN 注释掉后重新构建）
./rm_armor_vision.exe
# 传统方法：真值图误差评估
./eval_gt.exe gt/img1.jpg gt/img2.jpg gt/img3.jpg

# 神经网络方法：视频（默认 USE_NN 已启用）
./rm_armor_vision.exe
# 神经网络方法：端到端验证 + 真值图误差评估
../code/build/test_nn.exe                # 在项目根目录运行（内部按相对路径找图和模型）
./eval_gt_nn.exe gt/img1.jpg gt/img2.jpg gt/img3.jpg
```

结果图与日志：`eval_gt` 输出 `result_*.jpg`，`eval_gt_nn` 输出 `nn_result_*.jpg`（默认写在运行目录）。
