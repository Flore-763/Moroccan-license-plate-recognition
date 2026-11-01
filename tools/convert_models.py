"""
convert_models.py — convertit vos 2 modèles au format ONNX (exécutable dans le navigateur).

Usage (depuis la racine du dépôt) :
    pip install -r tools/requirements-convert.txt
    python tools/convert_models.py

Entrées  : models/plate_yolo.pt  et  models/crnn_best.keras
Sorties  : frontend/public/models/plate_yolo.onnx  et  crnn.onnx

Le CRNN est reconstruit directement en graphe ONNX (sans tf2onnx, qui ne supporte
pas Keras 3) puis VÉRIFIÉ numériquement contre le modèle Keras d'origine : le script
échoue si l'écart dépasse 1e-3.
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────── CRNN : Keras → ONNX ────────────────────────────
def _suffix(layer):
    m = re.search(r"_(\d+)$", layer.name)
    return int(m.group(1)) if m else 0


def _same_pads(size, k, s):
    """Padding « same » de TensorFlow (asymétrique quand stride=2)."""
    out = -(-size // s)
    total = max((out - 1) * s + k - size, 0)
    return total // 2, total - total // 2, out


def keras_crnn_to_onnx(model, out_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    L = model.layers
    convs = sorted([l for l in L if l.__class__.__name__ == "Conv2D"], key=_suffix)
    bns = sorted([l for l in L if l.__class__.__name__ == "BatchNormalization"], key=_suffix)
    leaky = [l for l in L if l.__class__.__name__ == "LeakyReLU"]
    bidir = next(l for l in L if l.__class__.__name__ == "Bidirectional")
    dense = [l for l in L if l.__class__.__name__ == "Dense"][-1]
    alpha = float(getattr(leaky[0], "negative_slope", getattr(leaky[0], "alpha", 0.3)))

    nodes, inits = [], []
    cnt = {"n": 0}

    def uid(p):
        cnt["n"] += 1
        return f"{p}_{cnt['n']}"

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name))
        return name

    def init_i64(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    def conv(x, layer, hw):
        kh, kw = layer.kernel_size
        sh, sw = layer.strides
        pt, pb, oh = _same_pads(hw[0], kh, sh)
        pl, pr, ow = _same_pads(hw[1], kw, sw)
        w = layer.get_weights()[0].transpose(3, 2, 0, 1)      # HWIO → OIHW
        y = uid("conv")
        nodes.append(helper.make_node(
            "Conv", [x, init(uid("W"), w)], [y],
            kernel_shape=[kh, kw], strides=[sh, sw], pads=[pt, pl, pb, pr]))
        return y, (oh, ow)

    def bn(x, layer):
        gamma, beta, mean, var = layer.get_weights()
        y = uid("bn")
        nodes.append(helper.make_node(
            "BatchNormalization",
            [x, init(uid("g"), gamma), init(uid("b"), beta), init(uid("m"), mean), init(uid("v"), var)],
            [y], epsilon=float(layer.epsilon)))
        return y

    def act(x):
        y = uid("lrelu")
        nodes.append(helper.make_node("LeakyRelu", [x], [y], alpha=alpha))
        return y

    in_h, in_w = model.input_shape[1], model.input_shape[2]
    # Entrée NHWC comme Keras (B,H,W,1) → NCHW, puis Rescaling 1/255
    x = "input"
    nodes.append(helper.make_node("Transpose", [x], ["nchw"], perm=[0, 3, 1, 2]))
    nodes.append(helper.make_node("Mul", ["nchw", init("scale", np.array(1 / 255.0))], ["x0"]))
    x, hw = "x0", (in_h, in_w)

    ci = bi = 0
    while ci < len(convs):                                   # un bloc résiduel par itération
        c1, c2 = convs[ci], convs[ci + 1]
        skip = None
        ci += 2
        if ci < len(convs) and convs[ci].kernel_size == (1, 1):
            skip, ci = convs[ci], ci + 1
        h, hw1 = conv(x, c1, hw)
        h = act(bn(h, bns[bi]))
        h, hw1 = conv(h, c2, hw1)
        h = bn(h, bns[bi + 1])
        bi += 2
        s = conv(x, skip, hw)[0] if skip is not None else x
        add = uid("add")
        nodes.append(helper.make_node("Add", [h, s], [add]))
        x, hw = act(add), hw1                                # Dropout = identité à l'inférence

    # (B,C,H,W) → Permute((2,1,3)) Keras ≡ (B,W,H,C) → Reshape (B,W,H*C)
    nodes.append(helper.make_node("Transpose", [x], ["seq_whc"], perm=[0, 3, 2, 1]))
    nodes.append(helper.make_node(
        "Reshape", ["seq_whc", init_i64("shape_seq", [0, 0, -1])], ["seq_bt"]))
    nodes.append(helper.make_node("Transpose", ["seq_bt"], ["seq_tb"], perm=[1, 0, 2]))  # (T,B,F)

    # BiLSTM : Keras (i,f,c,o) → ONNX (i,o,f,c)
    def pack(layer):
        kernel, rec, bias = layer.get_weights()
        u = rec.shape[0]
        def reorder(a):
            i, f, c, o = np.split(a, 4, axis=-1)
            return np.concatenate([i, o, f, c], axis=-1)
        return reorder(kernel).T, reorder(rec).T, np.concatenate([reorder(bias), np.zeros(4 * u)]), u

    Wf, Rf, Bf, units = pack(bidir.forward_layer)
    Wb, Rb, Bb, _ = pack(bidir.backward_layer)
    nodes.append(helper.make_node(
        "LSTM",
        ["seq_tb", init("lstm_W", np.stack([Wf, Wb])), init("lstm_R", np.stack([Rf, Rb])),
         init("lstm_B", np.stack([Bf, Bb]))],
        ["Y"], hidden_size=units, direction="bidirectional"))
    nodes.append(helper.make_node("Transpose", ["Y"], ["Y_btdh"], perm=[2, 0, 1, 3]))   # (B,T,2,H)
    nodes.append(helper.make_node(
        "Reshape", ["Y_btdh", init_i64("shape_y", [0, 0, -1])], ["lstm_out"]))

    kernel, bias = dense.get_weights()
    nodes.append(helper.make_node("MatMul", ["lstm_out", init("dense_W", kernel)], ["logits_nb"]))
    nodes.append(helper.make_node("Add", ["logits_nb", init("dense_b", bias)], ["logits"]))
    nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=-1))

    n_cls = kernel.shape[1]
    graph = helper.make_graph(
        nodes, "crnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_h, in_w, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "time", n_cls])],
        initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out_path))
    return m


def verify_crnn(keras_model, onnx_path, n=8, tol=1e-3):
    import onnxruntime as ort
    h, w = keras_model.input_shape[1:3]
    x = np.random.RandomState(0).randint(0, 256, (n, h, w, 1)).astype(np.float32)
    ref = keras_model.predict(x, verbose=0)
    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    same_argmax = bool((ref.argmax(-1) == got.argmax(-1)).all())
    print(f"[verif] écart max Keras↔ONNX = {diff:.2e} | argmax identiques : {same_argmax}")
    if diff > tol or not same_argmax:
        sys.exit("ÉCHEC : le CRNN ONNX ne reproduit pas le modèle Keras (voir README, section dépannage).")


def convert_crnn(keras_path, out_path):
    import keras
    spec = importlib.util.spec_from_file_location("model_02", ROOT / "model" / "CRNN" / "model_02.py")
    m02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m02)
    model = keras.models.load_model(str(keras_path), custom_objects={"CTCLoss": m02.CTCLoss}, compile=False)
    print(f"[crnn] entrée {model.input_shape} → sortie {model.output_shape}")
    keras_crnn_to_onnx(model, out_path)
    verify_crnn(model, out_path)
    print(f"[crnn] OK → {out_path} ({Path(out_path).stat().st_size/1024:.0f} Ko)")


# ─────────────────────────── YOLO : .pt → ONNX ──────────────────────────────
def convert_yolo(pt_path, out_path, imgsz=640):
    from ultralytics import YOLO
    produced = YOLO(str(pt_path)).export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(produced).replace(out_path)
    print(f"[yolo] OK → {out_path} ({out_path.stat().st_size/1e6:.1f} Mo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=ROOT / "models" / "plate_yolo.pt")
    ap.add_argument("--crnn", default=ROOT / "models" / "crnn_best.keras")
    ap.add_argument("--out", default=ROOT / "frontend" / "public" / "models")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--skip-crnn", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if not a.skip_crnn:
        convert_crnn(a.crnn, out / "crnn.onnx")
    if not a.skip_yolo:
        convert_yolo(Path(a.yolo), out / "plate_yolo.onnx")
