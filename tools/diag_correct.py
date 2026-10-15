"""用【正确】的通道解释解析 Infantry 模型：
   ch0..3  = 颜色(4类, sigmoid)
   ch4..12 = 装甲数字(9类 1-9, sigmoid)
   ch13..20 = 4个角点(x1,y1,x2,y2,x3,y3,x4,y4) 像素(相对640x480 letterbox)
"""
import sys, numpy as np
from PIL import Image
import onnxruntime as ort

MODEL = r"C:/work/electric_competition/robot/homework/second-train/code/models/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx"
IMGS = [
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img1.jpg",
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img2.jpg",
    r"C:/work/electric_competition/robot/homework/second-train/code/build/gt/img3.jpg",
]
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]; outname = sess.get_outputs()[0].name
H, W = 480, 640

def preprocess(path):
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
    h, w = img.shape[:2]; scale = min(W/w, H/h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    r = np.array(Image.fromarray(img).resize((nw, nh)))
    padX, padY = (W-nw)//2, (H-nh)//2
    sq = np.full((H, W, 3), 114, np.uint8); sq[padY:padY+nh, padX:padX+nw] = r
    blob = sq.astype(np.float32)[..., ::-1]/255.0
    return blob.transpose(2,0,1)[None,...], scale, padX, padY, (w,h)

for path in IMGS:
    name = path.split("/")[-1]
    blob, scale, padX, padY, (ow, oh) = preprocess(path)
    out = sess.run([outname], {inp.name: blob})[0][0]  # [21, 6300]
    color = out[0:4, :]      # [4, N]
    armor = out[4:13, :]     # [9, N]
    box = out[13:21, :]      # [8, N] corners
    colmx = color.max(axis=0); colid = color.argmax(axis=0)
    armx = armor.max(axis=0); armid = armor.argmax(axis=0)
    # 综合置信：颜色与数字都高
    score = colmx * armx
    # 过滤
    mask = (colmx > 0.25) & (armx > 0.25)
    idxs = np.where(mask)[0]
    print(f"\n=== {name}  检测候选(颜色>0.25 & 数字>0.25): {len(idxs)} 个 ===")
    if len(idxs) == 0:
        # 退而看颜色最高的
        o = np.argsort(-colmx)[:3]
        for i in o:
            print(f"  anchor {i}: color={colmx[i]:.3f}(cls{colid[i]}) armor={armx[i]:.3f}(d{armid[i]+1}) corners(px)=",
                  np.round(box[:, i], 1).tolist())
        continue
    order = idxs[np.argsort(-score[idxs])[:5]]
    for i in order:
        c = box[:, i].reshape(4, 2)  # 4 corners (x,y)
        # 还原到原图坐标
        c_img = (c - np.array([padX, padY])) / scale
        xs = c_img[:, 0]; ys = c_img[:, 1]
        cx, cy = xs.mean(), ys.mean()
        print(f"  anchor {i}: color={colmx[i]:.3f}(cls{colid[i]}) armor={armx[i]:.3f}(d{armid[i]+1}) score={score[i]:.3f}")
        print(f"      corners(px,letterbox)= {np.round(box[:,i],1).tolist()}")
        print(f"      corners(img)= {np.round(c_img,1).tolist()}  bbox~ x[{xs.min():.0f},{xs.max():.0f}] y[{ys.min():.0f},{ys.max():.0f}] center=({cx:.0f},{cy:.0f})")
