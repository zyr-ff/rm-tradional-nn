import sys, numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
IMG = r"C:/work/electric_competition/robot/homework/second-train/素材/图像/712c4f28e22091efdb528bf3294979fe.jpg"

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape, "type", inp.type)
outname = sess.get_outputs()[0].name
print("output:", outname, sess.get_outputs()[0].shape, "type", sess.get_outputs()[0].type)

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
blob = blob.transpose(2, 0,1)[None, ...]

try:
    out = sess.run([outname], {inp.name: blob})[0]
    print("fp16 out shape", out.shape, "dtype", out.dtype)
    out = out[0]
    print("box ch0 max|v|=%.4f  ch2 max|v|=%.4f" % (np.max(np.abs(out[0])), np.max(np.abs(out[2]))))
    cls_best = out[4:, :].max(axis=0)
    order = np.argsort(-cls_best)[:5]
    print("top-5 anchors by class logit (idx, cx,cy,w,h, logit):")
    for i in order:
        print("  ", int(i), np.round(out[0:4, i], 3), round(float(cls_best[i]), 3))
except Exception as e:
    print("RUN ERROR:", repr(e))
