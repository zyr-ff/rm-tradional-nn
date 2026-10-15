# 项目结构（按识别方法分离）

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
