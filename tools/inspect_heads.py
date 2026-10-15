"""读 Infantry 模型各 head 的 Conv 输出通道数 与 Reshape 目标形状，确定精确通道切分。"""
import onnx
from onnx import numpy_helper

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
m = onnx.load(MODEL)
g = m.graph

# 权重名 -> shape
w = {}
for init in g.initializer:
    w[init.name] = [d for d in init.dims]

def out_ch_of(conv_name):
    # 找 conv 节点的 weight 输入
    for n in g.node:
        if n.op_type == "Conv" and n.name == conv_name:
            wt = n.input[1]
            return w.get(wt, None)
    return None

# 每个尺度的最终 Conv（.2/Conv）输出通道
heads = {
    "cvcolor.0.2": "/model.22/cvcolor.0/cvcolor.0.2/Conv",
    "cvcolor.1.2": "/model.22/cvcolor.1/cvcolor.1.2/Conv",
    "cvcolor.2.2": "/model.22/cvcolor.2/cvcolor.2.2/Conv",
    "cvarmor.0.2": "/model.22/cvarmor.0/cvarmor.0.2/Conv",
    "cvarmor.1.2": "/model.22/cvarmor.1/cvarmor.1.2/Conv",
    "cvarmor.2.2": "/model.22/cvarmor.2/cvarmor.2.2/Conv",
    "cv4.0.2":     "/model.22/cv4.0/cv4.0.2/Conv",
    "cv4.1.2":     "/model.22/cv4.1/cv4.1.2/Conv",
    "cv4.2.2":     "/model.22/cv4.2/cv4.2.2/Conv",
}
print("=== 各 head 最终 Conv 输出通道 (权重 out_ch) ===")
for k, nm in heads.items():
    oc = out_ch_of(nm)
    print(f"  {k:14s} out_ch = {oc}")

# Reshape 目标常量
print("\n=== 关键 Reshape 的 target shape (Constant 输入) ===")
want = {
    "/model.22/Reshape_3": "/model.22/Constant_3",
    "/model.22/Reshape_6": "/model.22/Constant_6",
    "/model.22/Reshape_9": "/model.22/Constant_9",
    "/model.22/Reshape_10": "/model.22/Constant",
    "/model.22/Reshape": "/model.22/Constant",
}
# 找这些 Constant 的 dims
const_dims = {}
for init in g.initializer:
    if init.name in want.values():
        const_dims[init.name] = list(init.dims) + list(numpy_helper.to_array(init).flatten().tolist()[:8])
for reshape, const in want.items():
    print(f"  {reshape}  <- const {const} = {const_dims.get(const)}")

# 直接看 Concat_3 三个输入的"可推断"形状：从 Sigmoid/Mul 节点推
print("\n=== 直接统计每个 head 的 anchors 维（通过 Reshape 后的形状推断）===")
# 通过追踪每个 final conv -> reshape -> 看 constant
def find_reshape_target(reshape_name):
    for n in g.node:
        if n.name == reshape_name:
            const_name = n.input[1]
            for init in g.initializer:
                if init.name == const_name:
                    return numpy_helper.to_array(init).flatten().tolist()
    return None

for rn in ["/model.22/Reshape_3","/model.22/Reshape_6","/model.22/Reshape_9","/model.22/Reshape_10"]:
    t = find_reshape_target(rn)
    print(f"  {rn} target = {t}")
