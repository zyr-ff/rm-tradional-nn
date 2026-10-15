"""彻底诊断 Infantry fp16 模型输出格式。
对 3 张含装甲的仿真图跑推理，逐通道 dump，并找出框非零的 anchor。
"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
IMGS = [
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img1.jpg",
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img2.jpg",
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img3.jpg",
]

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
outname = sess.get_outputs()[0].name
print("input:", inp.name, inp.shape, "type", inp.type)
print("output:", outname, sess.get_outputs()[0].shape, "type", sess.get_outputs()[0].type)
print("providers:", sess.get_providers())

H, W = 480, 640

def preprocess(path):
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()  # RGB->BGR
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    r = np.array(Image.fromarray(img).resize((nw, nh)))
    padX, padY = (W - nw) // 2, (H - nh) // 2
    sq = np.full((H, W, 3), 114, np.uint8)
    sq[padY:padY+nh, padX:padX+nw] = r
    blob = sq.astype(np.float32)[..., ::-1] / 255.0   # BGR->RGB, /255
    blob = blob.transpose(2, 0, 1)[None, ...]          # NCHW
    return blob

for path in IMGS:
    print("\n" + "=" * 70)
    print("IMAGE:", path.split("/")[-1])
    blob = preprocess(path)
    out = sess.run([outname], {inp.name: blob})[0][0]  # [21, 6300]
    C_axis, N = out.shape
    print("out:", out.shape, "channels(4+C)=", C_axis, "=> numClasses=", C_axis - 4, " anchors=", N)

    # 逐通道全局最大绝对值 + 所在 anchor
    print("\n-- per-channel global max|v| and argmax --")
    for c in range(C_axis):
        amax = np.max(np.abs(out[c]))
        arg = int(np.argmax(np.abs(out[c])))
        print(f"  ch{c:2d} max|v|={amax:9.4f}  argmax@anchor={arg:5d}  rawmax={np.max(out[c]):9.4f}")

    # 找出框坐标明显非零的 anchor（假设 0..3 是框）
    print("\n-- anchors with plausible box (max(|box0..3|) > 0.1) --")
    box = out[:4, :]  # [4, N]
    boxmag = np.max(np.abs(box), axis=0)  # [N]
    idxs = np.where(boxmag > 0.1)[0]
    print("  count =", len(idxs))
    if len(idxs) > 0:
        # 按框幅值排序，取前 15
        order = idxs[np.argsort(-boxmag[idxs])[:15]]
        print("  anchor  cx      cy      w       h     | boxmag  clsMax  clsId  cls@4")
        for i in order:
            cx, cy, bw, bh = box[:, i]
            clsvec = out[4:, i]
            cid = int(np.argmax(clsvec))
            clsmax = float(clsvec[cid])
            print(f"  {i:5d} {cx:7.3f} {cy:7.3f} {bw:7.3f} {bh:7.3f} | {boxmag[i]:7.3f} {clsmax:9.2f} {cid:3d} {out[4,i]:9.2f}")

    # 反向：按类别最大值排序，看 top-15 的框是否非零
    print("\n-- top-15 anchors by class max (assuming ch4.. are class) --")
    cls = out[4:, :]
    clsmx = cls.max(axis=0)
    order = np.argsort(-clsmx)[:15]
    print("  anchor  cx      cy      w       h     | clsMax  clsId")
    for i in order:
        cx, cy, bw, bh = box[:, i]
        cid = int(cls[:, i].argmax())
        print(f"  {i:5d} {cx:7.3f} {cy:7.3f} {bw:7.3f} {bh:7.3f} | {clsmx[i]:9.2f} {cid:3d}")
