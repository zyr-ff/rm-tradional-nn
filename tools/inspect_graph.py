"""读 Infantry 模型计算图，定位 output0 的来源，看清楚 21 维怎么拼出来。"""
import onnx
from onnx import numpy_helper

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
m = onnx.load(MODEL)
g = m.graph
print("opset:", [op.version for op in m.opset_import])
print("inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output])

# 建名字->node 映射
name2node = {}
for n in g.node:
    for o in n.output:
        name2node[o] = n

# 从 output0 反向追溯
def trace(name, depth=0, seen=None):
    if seen is None: seen = set()
    if name in seen or depth > 12:
        return
    seen.add(name)
    if name in name2node:
        n = name2node[name]
        attr = ", ".join(f"{a.name}={onnx.helper.get_attribute_value(a)}" for a in n.attribute) if n.attribute else ""
        print("  " * depth + f"[{n.op_type}] {n.name}  out={list(n.output)}  in={list(n.input)}  {attr}")
        for inp in n.input:
            trace(inp, depth+1, seen)

print("\n=== trace output0 backward ===")
trace(g.output[0].name)

# 列出所有特殊后处理节点（可能有多个 output / 不同 head）
print("\n=== 所有 Sigmoid / Reshape / Concat / Transpose / Slice / Gather / Reduce 节点 ===")
for n in g.node:
    if n.op_type in ("Sigmoid", "Reshape", "Concat", "Transpose", "Slice", "Gather", "ReduceMean", "Softmax", "Mul", "Add", "Sub", "Div"):
        attr = ", ".join(f"{a.name}={onnx.helper.get_attribute_value(a)}" for a in n.attribute) if n.attribute else ""
        print(f"  [{n.op_type}] {n.name} in={list(n.input)} out={list(n.output)} {attr}")

# 输出所有节点名+op（前若干）
print("\n=== 全部节点 (op_type, name) 共", len(g.node), "个 ===")
for n in g.node:
    print(f"  [{n.op_type}] {n.name}")
