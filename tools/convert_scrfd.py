#!/usr/bin/env python3
"""Convert antelopev2 `scrfd_10g_bnkps.onnx` (SCRFD-10g face detector) -> safetensors for
the native MLX port (sc-3082), and dump golden raw outputs + authoritative insightface
detections for the parity test.

Architecture (verified by walking the onnx graph, opset 11; BatchNorm already folded into
biased convs):
  - **stem**: Conv(3->28,3x3,s2)+Relu, Conv(28->28,3x3,s1)+Relu, Conv(28->56,3x3,s1)+Relu, MaxPool2x2 s2
  - **backbone** stages, blocks [3,4,2,3] (channels 56/88/88/224). Each block:
    Conv(c1,3x3,stride)+Relu -> Conv(c2,3x3,s1) -> + identity -> Relu, where stage 2/3/4 block 0
    has stride 2 + a downsample (AvgPool2x2 s2 -> Conv1x1) on the identity; stage 1 has no downsample.
    Backbone taps: C2=stage2 (s8,88), C3=stage3 (s16,88), C4=stage4 (s32,224).
  - **neck (PAFPN)**: lateral 1x1 (->56) on C2/C3/C4; top-down nearest x2 upsample + add (P4,P3);
    fpn 3x3 convs; bottom-up downsample 3x3 s2 + add (N4,N5); pafpn 3x3 convs.
    Head inputs: s8 = fpn0(P3); s16 = pafpn0(N4); s32 = pafpn1(N5).
  - **heads** (per stride, NOT weight-shared): 3x Conv(3x3)+Relu (->80) then cls Conv(->2),
    reg Conv(->8) * learned scale, kps Conv(->20). Outputs reshaped to [-1,1]/[-1,4]/[-1,10];
    scores sigmoided; reg scaled (the per-level `bbox_head.scales.*` scalar) — all in-graph.

Conv weights onnx OIHW -> MLX OHWI. fp32. Outputs under tools/golden/ (gitignored):
  scrfd_10g.safetensors        -- converted weights + per-level reg scales (head{8,16,32}.scale)
  scrfd_goldens.safetensors    -- input [1,640,640,3] f32 (the 640 blob, NHWC), the 9 raw onnx
                                  outputs, and insightface detect boxes/kpss + det_scale on t1.jpg

Run with the dwpose-spike venv (onnx + onnxruntime + insightface + cv2 + numpy + safetensors):
  ~/.dwpose-spike/venv/bin/python tools/convert_scrfd.py
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

STAGE_BLOCKS = [3, 4, 2, 3]
SCRFD = os.path.expanduser("~/.insightface/models/antelopev2/scrfd_10g_bnkps.onnx")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "golden")
DET_SIZE = 640


def canonical_conv_names():
    """Canonical names for the 58 convs in onnx node order (matches the Rust module layout)."""
    names = ["stem.conv0", "stem.conv1", "stem.conv2"]
    for si, nb in enumerate(STAGE_BLOCKS, start=1):
        for b in range(nb):
            names.append(f"stage{si}.{b}.conv1")
            names.append(f"stage{si}.{b}.conv2")
            if b == 0 and si > 1:  # stage 1 block 0 has no downsample (s1, same channels)
                names.append(f"stage{si}.{b}.downsample")
    names += [
        "neck.lateral0", "neck.lateral1", "neck.lateral2",
        "neck.fpn0", "neck.fpn1", "neck.fpn2",
        "neck.down0", "neck.down1", "neck.pafpn0", "neck.pafpn1",
    ]
    for s in (8, 16, 32):
        names += [f"head{s}.stem0", f"head{s}.stem1", f"head{s}.stem2",
                  f"head{s}.cls", f"head{s}.reg", f"head{s}.kps"]
    return names


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    g = onnx.load(SCRFD).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    def f32(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

    out = {}
    conv_nodes = [n for n in g.node if n.op_type == "Conv"]
    names = canonical_conv_names()
    assert len(conv_nodes) == len(names), f"{len(conv_nodes)} convs vs {len(names)} names"
    for name, n in zip(names, conv_nodes):
        out[f"{name}.weight"] = f32(np.transpose(init[n.input[1]], (0, 2, 3, 1)))  # OIHW -> OHWI
        out[f"{name}.bias"] = f32(init[n.input[2]])

    # per-level reg scales (in-graph Mul) -> head{8,16,32}.scale
    for s, key in zip((8, 16, 32), ("bbox_head.scales.0.scale", "bbox_head.scales.1.scale", "bbox_head.scales.2.scale")):
        out[f"head{s}.scale"] = f32(np.asarray(init[key]).reshape(()))

    print(f"converted {len(out)} tensors")
    for k in ("stem.conv0.weight", "stage2.0.downsample.weight", "neck.lateral2.weight",
              "neck.pafpn1.weight", "head8.cls.weight", "head8.reg.weight", "head8.kps.weight",
              "head8.scale", "head16.scale", "head32.scale"):
        print(f"  {k:26} {tuple(out[k].shape)}  {out[k].ravel()[:1] if out[k].ndim==0 or out[k].size==1 else ''}")

    from safetensors.numpy import save_file
    wpath = os.path.join(OUT_DIR, "scrfd_10g.safetensors")
    save_file(out, wpath)
    print("wrote", wpath)

    # --- goldens: real face image, insightface-identical preprocessing + authoritative detect
    import cv2
    import insightface
    import onnxruntime as ort

    img_path = os.path.join(os.path.dirname(insightface.__file__), "data", "images", "t1.jpg")
    img = cv2.imread(img_path)  # BGR
    h, w = img.shape[:2]
    im_ratio = float(h) / w
    if im_ratio > 1.0:
        new_h, new_w = DET_SIZE, int(DET_SIZE / im_ratio)
    else:
        new_w, new_h = DET_SIZE, int(DET_SIZE * im_ratio)
    det_scale = float(new_h) / h
    resized = cv2.resize(img, (new_w, new_h))
    det_img = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized
    # blobFromImage: (BGR - 127.5)/128 with swapRB -> RGB, NCHW [1,3,640,640]
    blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True)

    sess = ort.InferenceSession(SCRFD, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    raw = sess.run(out_names, {in_name: blob})
    # outputs ordered: 3 scores, 3 bbox, 3 kps (strides 8,16,32) per graph output order
    print("raw output shapes:", [r.shape for r in raw])

    goldens = {"input": f32(np.transpose(blob, (0, 2, 3, 1)))}  # NHWC for MLX
    for s, sc, bb, kp in zip((8, 16, 32), raw[0:3], raw[3:6], raw[6:9]):
        goldens[f"score.{s}"] = f32(sc)
        goldens[f"bbox.{s}"] = f32(bb)
        goldens[f"kps.{s}"] = f32(kp)

    det = insightface.model_zoo.get_model(SCRFD, providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=0, input_size=(DET_SIZE, DET_SIZE))
    det.det_thresh = 0.5
    bboxes, kpss = det.detect(img, input_size=(DET_SIZE, DET_SIZE))
    print(f"insightface detected {bboxes.shape[0]} faces; scores {bboxes[:,4].round(3).tolist()}")
    goldens["det_bboxes"] = f32(bboxes)  # [M,5] x1,y1,x2,y2,score (original-image coords)
    goldens["det_kpss"] = f32(kpss)      # [M,5,2]
    goldens["det_scale"] = f32(np.asarray(det_scale).reshape(()))

    from safetensors.numpy import save_file as save2
    gpath = os.path.join(OUT_DIR, "scrfd_goldens.safetensors")
    save2(goldens, gpath)
    print("wrote", gpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
