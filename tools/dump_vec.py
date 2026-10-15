"""dump 关键 anchor 的完整 21 维向量，确定真实通道布局。"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
IMG = r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img1.jpg"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
outname = sess.get_outputs()[0].name
H, W = 480, 640

img = np.array(Image.open(IMG).convert("RGB"))[:, :, ::-1].copy()
h, w = img.shape[:2]
scale = min(W / w, H / h)
nw, nh = int(round(w * scale)), int(round(h * scale))
r = np.array(Image.fromarray(img).resize((nw, nh)))
padX, padY = (W - nw) // 2, (H - nh) // 2
sq = np.full((H, W, 3), 114, np.uint8)
sq[padY:padY+nh, padX:padX+nw] = r
blob = sq.astype(np.float32)[..., ::-1] / 255.0
blob = blob.transpose(2, 0, 1)[None, ...]

out = sess.run([outname], {inp.name: blob})[0][0]  # [21, 6300]

# 选几个关键 anchor
cands = [2681, 2682, 2676, 4473, 4472, 4553, 5839]
print("anchor : full 21-vector (per channel)")
for a in cands:
    v = out[:, a]
    s = " ".join(f"{x:8.3f}" for x in v)
    print(f"{a:5d} : {s}")

print("\n--- 把 21 通道按值大小排序，看 anchor 2681 的非零通道 ---")
for a in [2681, 4473]:
    v = out[:, a]
    order = np.argsort(-np.abs(v))[:8]
    print(f"anchor {a}: top nonzero channels ->", [(int(c), round(float(v[c]),3)) for c in order])

# 统计：每个通道上有多少个 anchor 的绝对值 > 0.05
print("\n--- 每通道 '活跃' anchor 数 (|v|>0.05) ---")
for c in range(21):
    n = int(np.sum(np.abs(out[c]) > 0.05))
    print(f"  ch{c:2d}: {n:5d}  max={np.max(np.abs(out[c])):9.3f}")
