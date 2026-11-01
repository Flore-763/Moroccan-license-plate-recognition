"""Génère frontend/tools-ref.json (références Python) ; puis : cd frontend && npm test  — vérifie que le JS = le Python."""
import sys, json, re, importlib.util, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, cv2
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location("inf",R+"/model/CRNN/04_inference.py"); inf=importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)
rng=np.random.RandomState(3)
ref={"resize":[],"ctc":[],"plates":[]}
# 1) resize cubique vs cv2 (images aléatoires lissées pour ressembler à une plaque)
for (w,h) in [(300,90),(181,57),(60,20),(128,32),(517,140)]:
    g=cv2.GaussianBlur(rng.randint(0,256,(h,w)).astype(np.uint8),(0,0),1.5)
    out=cv2.resize(g,(128,32),interpolation=cv2.INTER_CUBIC)
    ref["resize"].append({"w":w,"h":h,"src":g.flatten().tolist(),"out":out.flatten().tolist()})
# 2) CTC + confiance sur des sorties softmax aléatoires "pointues"
for _ in range(6):
    T,C=32,16
    logits=rng.normal(0,1,(T,C))*3
    p=np.exp(logits); p/=p.sum(1,keepdims=True)
    txt=inf.decode_ctc(p[None])[0]
    ref["ctc"].append({"p":p.flatten().tolist(),"text":txt,"conf":float(p.max(-1).mean())})
# 3) validation de plaque
def validate(raw, conf):
    text=inf.normalize_label(raw)
    if not text: return None
    m=inf._PLATE_REGEX.search(text)
    if m: return [f"{m.group(1)} {m.group(2)} {m.group(3)}", conf]
    d=re.sub(r"[^0-9]","",text)
    if len(d)>=3: return [text, conf*0.7]
    return None
for raw in ["76970ب40","4631ب40","82858ب1","12345ه7","999","12","ب","5678","123456أ12","90120أ72","7ب40"]:
    ref["plates"].append({"raw":raw,"conf":0.9,"out":validate(raw,0.9)})
json.dump(ref,open(R+"/frontend/tools-ref.json","w"))
print("réf OK", [r["text"] for r in ref["ctc"]])
