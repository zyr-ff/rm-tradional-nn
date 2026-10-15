"""把 fp16 ONNX 模型转换为 fp32，使其能被 OpenCV 的 dnn 解析器加载。
原因：Infantry-v8n-fp16 带了 fp16->fp32 的 Cast 节点 + fp16 权重，
OpenCV 5.0 的 ONNX 解析器不支持，会报错。转成 fp32 后数值几乎无损。
用法：python fp16_to_fp32.py <in.onnx> <out.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper

def main():
    if len(sys.argv) != 3:
        print("用法: python fp16_to_fp32.py <in.onnx> <out.onnx>")
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]
    model = onnx.load(in_path)
    graph = model.graph

    # 1) 把所有 fp16 的 initializer（权重）转成 fp32
    cnt = 0
    for init in graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_init)
            cnt += 1
    print(f"转换 fp16 权重: {cnt} 个")

    # 2) 把 graph 的 input/output/value_info 的 elem_type 由 fp16 -> fp32
    def fix_type(t):
        if t.elem_type == onnx.TensorProto.FLOAT16:
            t.elem_type = onnx.TensorProto.FLOAT
    for v in graph.input:
        fix_type(v.type.tensor_type)
    for v in graph.output:
        fix_type(v.type.tensor_type)
    for v in graph.value_info:
        fix_type(v.type.tensor_type)

    # 3) 移除所有 Cast 节点（现在全图都是 fp32，Cast 多余），并把引用重连到 Cast 的输入
    cast_map = {}          # cast 输出名 -> cast 输入名
    new_nodes = []
    removed = 0
    for node in graph.node:
        if node.op_type == "Cast":
            cast_map[node.output[0]] = node.input[0]
            removed += 1
            continue
        new_nodes.append(node)
    for node in new_nodes:
        for i in range(len(node.input)):
            if node.input[i] in cast_map:
                node.input[i] = cast_map[node.input[i]]
    # graph 的输出如果曾是某个 Cast 的输出，也要重连到 Cast 的输入
    for out in graph.output:
        if out.name in cast_map:
            out.name = cast_map[out.name]
    del graph.node[:]
    graph.node.extend(new_nodes)
    print(f"移除 Cast 节点: {removed} 个")

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"已保存 fp32 模型: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
