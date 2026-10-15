"""用 onnxruntime 跑 Infantry 模型，诊断输出格式，与 C++ 解析对照。"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp32.onnx"
IMG = sys.argv[1] if len(sys.argv) > 1 else r"C:/work/electric_competition/robot/homework/second-train/素材/图像/712c4f28e22091efdb528bf3294979fe.jpg"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inname = sess.get_inputs()[0].name
inshape = sess.get_inputs()[0].shape
print("input name:", inname, "shape:", inshape)
outname = sess.get_outputs()[0].name
print("output name:", outname, "shape:", sess.get_outputs()[0].shape)

# 预处理：letterbox 到 480x640（与 C++ 一致），BGR->RGB，/255
H, W = 480, 640
img = Image.open(IMG).convert("RGB")
img = np.array(img)[:, :, ::-1].copy()            # RGB -> BGR（模拟 OpenCV 读入）
h, w = img.shape[:2]
scale = min(W / w, H / h)
nw, nh = int(round(w * scale)), int(round(h * scale))
r = np.array(Image.fromarray(img).resize((nw, nh)))
padX, padY = (W - nw) // 2, (H - nh) // 2
sq = np.full((H, W, 3), 114, dtype=np.uint8)
sq[padY:padY+nh, padX:padX+nw] = r
blob = sq.astype(np.float32)[..., ::-1] / 255.0   # BGR->RGB, /255
blob = blob.transpose(2, 0, 1)[None, ...]         # NCHW

out = sess.run([outname], {inname: blob})[0]      # [1, 21, 6300]
print("out shape:", out.shape)
out = out[0]  # [21, 6300]
C_axis, N = out.shape
print("C_axis(4+C)=", C_axis, "N(anchors)=", N, "=> numClasses =", C_axis - 4)

# 4 个框通道的最大绝对值
for c in range(4):
    print(f"  box ch{c} max|v| = {np.max(np.abs(out[c])):.4f}  min={np.min(out[c]):.4f} max={np.max(out[c]):.4f}")

# 逐 anchor：cx,cy,w,h + 最佳类别（原始 logit）
boxes4 = out[:4, :]            # [4, N]
cls = out[4:, :]               # [17, N]
cls_best = cls.max(axis=0)     # 每 anchor 最佳类别 logit
order = np.argsort(-cls_best)[:8]
print("\nTop-8 anchors (by class logit), raw values:")
print("  idx  cx      cy      w       h       bestClsLogit  clsId")
for i in order:
    cx, cy, bw, bh = boxes4[:, i]
    cid = int(cls[:, i].argmax())
    print(f"  {i:4d}  {cx:7.2f} {cy:7.2f} {bw:7.2f} {bh:7.2f}  {cls_best[i]:12.3f}  {cid}")

# 尝试标准 YOLOv8 后处理：类别过 sigmoid，框当像素坐标
print("\n--- 标准后处理(类别sigmoid, 框当像素) Top-8 ---")
cls_sig = 1.0 / (1.0 + np.exp(-cls))
conf = cls_sig.max(axis=0)
order2 = np.argsort(-conf)[:8]
for i in order2:
    cx, cy, bw, bh = boxes4[:, i]
    cid = int(cls_sig[:, i].argmax())
    print(f"  {i:4d}  cx={cx:7.2f} cy={cy:7.2f} w={bw:7.2f} h={bh:7.2f} conf={conf[i]:.3f} cls={cid}")
