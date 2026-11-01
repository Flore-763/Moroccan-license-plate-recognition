import { useState, useEffect, useCallback, useRef } from "react";

// Dev : backend local. Prod (Docker) : même origine → "". Front séparé : VITE_API_URL.
const API = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
const POLL_MS = 3000;
// Mode statique (GitHub Pages…) : tout tourne dans le navigateur, pas de serveur.
const STATIC = import.meta.env.VITE_STATIC === "1";
const BASE = import.meta.env.BASE_URL;
const assetUrl = (u) => (STATIC ? BASE + u : API + u);
const STATIC_CONFIG = { demo_mode: true, persist_detections: false, enable_video: false,
  enable_webcam: false, max_image_mb: 15, static: true };
// Intervalle webcam : envoie 1 frame toutes les 600ms au backend
const WEBCAM_INTERVAL_MS = 150;
// Cooldown frontend : ne pas re-soumettre une zone pendant N envois

// Grille de zones pour la déduplication webcam (8×8)

const STATUS = {
  safe:       { label: "SAFE",    bg: "#f0fdf4", fg: "#15803d", dot: "#22c55e", border: "#bbf7d0" },
  vole:       { label: "VOLÉ",    bg: "#fef2f2", fg: "#b91c1c", dot: "#ef4444", border: "#fecaca" },
  suspicieux: { label: "SUSPECT", bg: "#fff7ed", fg: "#c2410c", dot: "#f97316", border: "#fed7aa" },
  inconnu:    { label: "INCONNU", bg: "#f5f3ff", fg: "#6d28d9", dot: "#8b5cf6", border: "#ddd6fe" },
};

const C = {
  bg: "#f1f5f9", surface: "#ffffff", surface2: "#f8fafc",
  border: "#e2e8f0", border2: "#cbd5e1",
  text: "#0f172a", textSub: "#475569", textMute: "#94a3b8",
  accent: "#2563eb", accentBg: "#eff6ff",
  header: "#1e3a5f",
};

const parseDate = (iso) =>
  new Date(typeof iso === "string" && !/(Z|[+-]\d{2}:?\d{2})$/.test(iso) ? iso + "Z" : iso);

const fmt = (iso) => {
  const d = parseDate(iso);
  return d.toLocaleDateString("fr-MA", { day: "2-digit", month: "short" }) + " " +
         d.toLocaleTimeString("fr-MA", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

async function apiFetch(path) {
  if (STATIC) {
    if (path === "/config") return STATIC_CONFIG;
    if (path === "/vehicules") return (await fetch(BASE + "data/vehicules.json")).json();
    return [];                                   // /detections, /alerts : historique local seulement
  }
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Composants partagés ───────────────────────────────────────────────────────

function StatusBadge({ statut, size = "sm" }) {
  const s = STATUS[statut] || STATUS.inconnu;
  return (
    <span style={{ background: s.bg, color: s.fg, border: `1px solid ${s.border}`, borderRadius: 5,
      padding: size === "lg" ? "5px 12px" : "2px 9px", fontSize: size === "lg" ? "0.74rem" : "0.64rem",
      fontWeight: 700, letterSpacing: "0.06em", whiteSpace: "nowrap",
      display: "inline-flex", alignItems: "center", gap: 5 }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.dot,
        display: "inline-block", flexShrink: 0, boxShadow: `0 0 4px ${s.dot}88` }} />
      {s.label}
    </span>
  );
}


function Plate({ text }) {
  return (
    <span style={{ fontFamily: "'Courier New', monospace", fontSize: "0.95rem", fontWeight: 800,
      background: "#1e3a5f", color: "#ffffff", border: "2px solid #c8a94a",
      borderRadius: 5, padding: "3px 12px", letterSpacing: "0.14em",
      whiteSpace: "nowrap", boxShadow: "0 1px 4px #0002", direction: "ltr" }}>
      {/* Chaque segment est un inline-block : le sens d'écriture arabe ne réordonne plus les chiffres */}
      {String(text ?? "").split(/(\d+)/).filter(Boolean).map((part, i) => (
        <span key={i} style={{ display: "inline-block", margin: "0 0.12em" }}>{part}</span>
      ))}
    </span>
  );
}

function StatCard({ label, value, accent, sub }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderTop: `3px solid ${accent}`,
      borderRadius: 10, padding: "18px 22px", display: "flex", flexDirection: "column", gap: 4,
      boxShadow: "0 1px 6px #0000000a" }}>
      <span style={{ color: C.textMute, fontSize: "0.67rem", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
      <span style={{ color: accent, fontSize: "1.85rem", fontWeight: 800, lineHeight: 1 }}>{value}</span>
      {sub && <span style={{ color: C.textMute, fontSize: "0.68rem" }}>{sub}</span>}
    </div>
  );
}

function SectionHeader({ children, count }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <h2 style={{ margin: 0, fontSize: "0.78rem", fontWeight: 700, color: C.textSub,
        textTransform: "uppercase", letterSpacing: "0.12em" }}>{children}</h2>
      {count !== undefined && (
        <span style={{ background: C.accentBg, color: C.accent, borderRadius: 99,
          padding: "1px 9px", fontSize: "0.68rem", fontWeight: 600 }}>{count}</span>
      )}
    </div>
  );
}

// ── Carte résultat plaque ─────────────────────────────────────────────────────




// ── Carte résultat d'une détection plaque ─────────────────────────────────────
function PlateResultCard({ item, showTrack = false }) {
  const s = STATUS[item.statut] || STATUS.inconnu;
  return (
    <div style={{ background: s.bg, border: `1px solid ${s.border}`,
      borderLeft: `4px solid ${s.dot}`, borderRadius: 8,
      padding: "11px 14px", boxShadow: "0 1px 4px #0000000a" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <StatusBadge statut={item.statut} size="lg" />
        <Plate text={item.immatriculation} />
        <span style={{ color: item.confiance > 0.8 ? "#15803d"
                            : item.confiance > 0.6 ? "#c2410c" : "#b91c1c",
          fontWeight: 700, fontSize: "0.85rem" }}>
          {(item.confiance * 100).toFixed(1)}%
        </span>
        {item.alerte_creee && (
          <span style={{ background: "#fef2f2", color: "#b91c1c",
            border: "1px solid #fecaca", borderRadius: 4,
            padding: "2px 7px", fontSize: "0.62rem", fontWeight: 700 }}>
            ⚠ ALERTE
          </span>
        )}
      </div>
      {/* Méta-données ByteTrack */}
      <div style={{ display: "flex", gap: 14, marginTop: 7, flexWrap: "wrap" }}>
        {showTrack && item.plate_track_id != null && (
          <span style={{ background: C.accentBg, color: C.accent,
            borderRadius: 4, padding: "1px 7px", fontSize: "0.65rem", fontWeight: 700 }}>
            {/* 🏷*/} Track #{item.plate_track_id} 
          </span>
        )}
        {showTrack && item.car_track_id != null && (
          <span style={{ background: "#fff7ed", color: "#c2410c",
            borderRadius: 4, padding: "1px 7px", fontSize: "0.65rem", fontWeight: 600 }}>
            🚗 Track #{item.car_track_id}
          </span>
        )}
        {item.votes != null && (
          <span style={{ background: C.surface2, color: C.textSub,
            borderRadius: 4, padding: "1px 7px", fontSize: "0.65rem" }}>
            🗳 {item.votes} vote{item.votes > 1 ? "s" : ""}
          </span>
        )}
        {item.timestamp_fmt && (
          <span style={{ color: C.textMute, fontSize: "0.68rem" }}>
            ⏱ {item.timestamp_fmt}
          </span>
        )}
        {item.frame && (
          <span style={{ color: C.textMute, fontSize: "0.68rem" }}>
            📸 Frame {item.frame}
          </span>
        )}
        {item.sharpness != null && (
          <span style={{ color: C.textMute, fontSize: "0.68rem" }}>
            🔍 Netteté {item.sharpness.toFixed(0)}
          </span>
        )}
        <span style={{ color: item.vehicule_trouve ? "#15803d" : "#6d28d9",
          fontSize: "0.68rem", fontWeight: 600, marginLeft: "auto" }}>
          {item.vehicule_trouve ? "✓ En DB" : "Non répertorié"}
        </span>
      </div>
    </div>
  );
}


// ── Navigation par onglets ────────────────────────────────────────────────────
const TABS = [
  { id: "upload",     label: "📷 Image"         },
  { id: "dashboard",  label: "Tableau de bord" },
  { id: "video",      label: "🎬 Vidéo"         },
  { id: "detections", label: "Détections"        },
  { id: "alerts",     label: "Alertes"           },
  { id: "vehicules",  label: "Véhicules"         },
];

function TabBar({ active, onChange, alertCount, tabs = TABS }) {
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: `2px solid ${C.border}`,
      marginBottom: 28, flexWrap: "wrap", background: C.surface,
      borderRadius: "10px 10px 0 0", padding: "0 8px",
      boxShadow: "0 1px 4px #0000000a" }}>
      {tabs.map(t => {
        const isActive = active === t.id;
        return (
          <button key={t.id} onClick={() => onChange(t.id)} style={{
            background: "transparent", border: "none",
            borderBottom: isActive ? `2px solid ${C.accent}` : "2px solid transparent",
            color: isActive ? C.accent : C.textSub,
            padding: "12px 18px", cursor: "pointer",
            fontSize: "0.8rem", fontWeight: isActive ? 700 : 500,
            transition: "all .15s", position: "relative", marginBottom: "-2px" }}>
            {t.label}
            {t.id === "alerts" && alertCount > 0 && (
              <span style={{ position: "absolute", top: 7, right: 4,
                background: "#ef4444", color: "#fff", borderRadius: 99,
                fontSize: "0.52rem", padding: "1px 5px", fontWeight: 700 }}>
                {alertCount}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}


// ════════════════════════════════════════════════════════════════════════════
//  DETECTION VIEW  —  image / fichier vidéo / webcam en direct
// ════════════════════════════════════════════════════════════════════════════
//
//  Un seul onglet, 3 modes sélectionnables :
//    "image"  → POST /detect-image   (photo)
//    "file"   → POST /detect-video   (fichier mp4/avi/…)  ByteTrack
//    "webcam" → POST /detect-frame   (flux webcam)        ByteTrack live
//

// À ajouter après le composant DetectionView ou à remplacer
// État du chargement des modèles ONNX (mode statique)
let _ready = false;
const modelsReady = () => _ready;

function ImageView({ onDetectionDone, maxMb = 8 }) {
  const [samples, setSamples] = useState([]);
  useEffect(() => {
    fetch(STATIC ? `${BASE}samples/samples.json` : `${API}/samples`)
      .then(r => (r.ok ? r.json() : [])).then(setSamples).catch(() => {});
  }, []);
  const [file, setFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const reset = () => {
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(null);
    setFileURL(null);
    setResult(null);
    setError(null);
  };

  const loadFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Seules les images sont acceptées (JPG/PNG/WEBP)");
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setError(`Image trop lourde (max ${maxMb} Mo)`);
      return;
    }
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(f);
    setFileURL(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const useSample = async (sample) => {
    try {
      const blob = await (await fetch(assetUrl(sample.url))).blob();
      const ext = (sample.url.split(".").pop() || "jpg").toLowerCase();
      const f = new File([blob], `${sample.name}.${ext}`, { type: blob.type || "image/jpeg" });
      loadFile(f);
      submitImage(f);
    } catch { setError("Impossible de charger l'image d'exemple."); }
  };

  const submitImage = async (f = file) => {
    if (!f) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let data;
      if (STATIC) {
        data = await (await import("./lib/localDetector.js")).detectImageLocal(f);
      } else {
        const fd = new FormData();
        fd.append("file", f);
        const r = await fetch(`${API}/detect-image`, { method: "POST", body: fd });
        data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      }
      setResult({ type: "image", ...data });
      _ready = true;
      if (data.success && onDetectionDone) onDetectionDone(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const ss = STATUS[result?.statut] || STATUS.inconnu;

  return (
    <div style={{ maxWidth: 760, margin: "0 auto" }}>
      {!file && samples.length > 0 && (
        <div style={{ marginBottom: 22 }}>
          <p style={{ color: C.textSub, fontSize: "0.78rem", fontWeight: 700, margin: "0 0 10px" }}>
            Essayez avec une image d'exemple :
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {samples.map(sm => (
              <button key={sm.url} onClick={() => useSample(sm)} title={sm.name}
                style={{ padding: 0, border: `2px solid ${C.border}`, borderRadius: 10,
                  overflow: "hidden", cursor: "pointer", background: C.surface,
                  width: 130, textAlign: "left" }}>
                <img src={assetUrl(sm.url)} alt={sm.name} loading="lazy"
                  style={{ width: "100%", height: 86, objectFit: "cover", display: "block" }} />
                <span style={{ display: "block", fontSize: "0.66rem", color: C.textSub,
                  padding: "5px 8px", whiteSpace: "nowrap", overflow: "hidden",
                  textOverflow: "ellipsis" }}>{sm.name}</span>
              </button>
            ))}
          </div>
          <p style={{ color: C.textMute, fontSize: "0.68rem", margin: "14px 0 0" }}>
            …ou importez votre propre photo ci-dessous.
          </p>
        </div>
      )}
      {!file ? (
        <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); loadFile(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current.click()}
          style={{ border: `2px dashed ${dragging ? C.accent : C.border2}`,
            borderRadius: 14, padding: "56px 24px", textAlign: "center",
            cursor: "pointer", background: dragging ? C.accentBg : C.surface,
            transition: "all .2s", boxShadow: "0 1px 4px #0000000a" }}>
          <div style={{ width: 56, height: 56, borderRadius: "50%",
            background: C.accentBg, border: `1px solid ${C.border}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "1.6rem", margin: "0 auto 16px" }}>📷</div>
          <p style={{ color: C.text, margin: 0, fontSize: "0.9rem", fontWeight: 600 }}>
            Glissez une photo ici ou{" "}
            <span style={{ color: C.accent, textDecoration: "underline" }}>cliquez pour parcourir</span>
          </p>
          <p style={{ color: C.textMute, margin: "8px 0 0", fontSize: "0.72rem" }}>JPG · PNG · WEBP</p>
          <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
            onChange={(e) => loadFile(e.target.files[0])} />
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <div>
            <div style={{ position: "relative" }}>
              <div style={{ position: "relative", display: "inline-block", width: "100%" }}>
                <img src={fileURL} alt="aperçu" style={{ width: "100%", borderRadius: 10,
                  border: `1px solid ${C.border}`, maxHeight: 340, objectFit: "contain",
                  background: C.surface2, display: "block" }} />
                {result?.bbox_plaque && result?.image_size && (
                  <svg viewBox={`0 0 ${result.image_size[0]} ${result.image_size[1]}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{ position: "absolute", inset: 0, width: "100%", height: "100%",
                      pointerEvents: "none" }}>
                    <rect x={result.bbox_plaque[0]} y={result.bbox_plaque[1]}
                      width={result.bbox_plaque[2] - result.bbox_plaque[0]}
                      height={result.bbox_plaque[3] - result.bbox_plaque[1]}
                      fill="none" stroke={ss.dot} strokeWidth={Math.max(3, result.image_size[0] / 250)} />
                  </svg>
                )}
              </div>
              <button onClick={reset} style={{ position: "absolute", top: 8, right: 8,
                background: C.surface, border: `1px solid ${C.border2}`, borderRadius: 6,
                color: C.textSub, cursor: "pointer", padding: "4px 10px", fontSize: "0.7rem" }}>
                ✕
              </button>
            </div>
            <p style={{ color: C.textMute, fontSize: "0.7rem", margin: "8px 0 0" }}>
              {file.name} — {(file.size / 1024).toFixed(0)} Ko
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {!result && !loading && (
              <button onClick={() => submitImage()} style={{ background: C.accent, border: "none",
                borderRadius: 8, color: "#fff", fontWeight: 700, fontSize: "0.88rem",
                padding: "13px 0", cursor: "pointer", boxShadow: `0 4px 14px ${C.accent}44` }}>
                Analyser la plaque →
              </button>
            )}
            {loading && <Spinner label={STATIC && !modelsReady()
              ? "Chargement des modèles (≈ 15 Mo, première fois seulement)…" : "Analyse en cours…"} />}
            {error && <ErrorBox msg={error} />}
            {result && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ background: ss.bg, border: `1px solid ${ss.border}`,
                  borderRadius: 10, padding: "16px 18px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                    <StatusBadge statut={result.statut || "inconnu"} size="lg" />
                    {result.alerte_creee && <AlertTag />}
                  </div>
                  {result.success ? (
                    <>
                      <span style={{ color: C.textMute, fontSize: "0.65rem", display: "block", marginBottom: 5 }}>IMMATRICULATION</span>
                      <Plate text={result.immatriculation} />
                      {result.plaque_image && (
                        <img src={result.plaque_image} alt="plaque détectée"
                          style={{ display: "block", marginTop: 12, maxWidth: "100%", maxHeight: 70,
                            borderRadius: 6, border: `1px solid ${C.border2}`, background: "#fff" }} />
                      )}
                      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
                        <ConfScore conf={result.confiance} />
                        <DBStatus found={result.vehicule_trouve} />
                      </div>
                      {result.statut === "inconnu" && (
                        <p style={{ color: C.textMute, fontSize: "0.66rem", margin: "10px 0 0" }}>
                          « Inconnu » = plaque absente de la base de démonstration.
                        </p>
                      )}
                      {result.temps_ms != null && (
                        <p style={{ color: C.textMute, fontSize: "0.64rem", margin: "6px 0 0" }}>
                          Analysé en {(result.temps_ms / 1000).toFixed(1)} s
                        </p>
                      )}
                    </>
                  ) : (
                    <p style={{ color: C.textSub, fontSize: "0.82rem", margin: 0 }}>
                      {result.message}. Essayez une photo où la plaque est bien visible, de face.
                    </p>
                  )}
                </div>
                <button onClick={reset} style={{ background: C.surface,
                  border: `1px solid ${C.border2}`, borderRadius: 8, color: C.textSub,
                  cursor: "pointer", padding: "9px 0", fontSize: "0.76rem", fontWeight: 600 }}>
                  Analyser une autre image
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


// À ajouter après ImageView
function VideoView({ onDetectionDone, enableWebcam = true }) {
  const [mode, setMode] = useState("file"); // "file" ou "webcam"
  const [file, setFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  // Webcam states
  const [camRunning, setCamRunning] = useState(false);
  const [camError, setCamError] = useState(null);
  const [liveVisible, setLiveVisible] = useState([]);
  const [liveDetect, setLiveDetect] = useState([]);
  const [sending, setSending] = useState(false);
  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayRef = useRef();
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const overlayBoxes = useRef([]);
  const rafRef = useRef(null);

  const reset = () => {
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(null);
    setFileURL(null);
    setResult(null);
    setError(null);
    setProgress(null);
  };

  const loadFile = (f) => {
    if (!f) return;
    if (!f.type.startsWith("video/") && !/\.(mp4|avi|mov|mkv|webm)$/i.test(f.name)) {
      setError("Vidéo attendue (MP4/AVI/MOV/MKV/WEBM)");
      return;
    }
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(f);
    setFileURL(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const submitVideo = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress("Envoi au serveur…");
    const steps = [
      "ByteTrack — suivi des plaques…",
      "Sélection du frame le plus net…",
      "Lecture CRNN (1 fois par véhicule)…",
      "Vérification en base de données…",
    ];
    let tick = 0;
    const timer = setInterval(() => {
      tick = Math.min(tick + 1, steps.length - 1);
      setProgress(steps[tick]);
    }, 4000);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/detect-video`, { method: "POST", body: fd });
      clearInterval(timer);
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      setResult({ type: "video", ...data });
      if ((data.plates_count || 0) > 0 && onDetectionDone) onDetectionDone();
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(timer);
      setLoading(false);
      setProgress(null);
    }
  };

  // Webcam functions
  const startWebcam = async () => {
    setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCamRunning(true);
      setLiveVisible([]);
      setLiveDetect([]);
      overlayBoxes.current = [];
      await fetch(`${API}/reset-camera`, { method: "POST", body: new FormData() });
      startFrameLoop();
    } catch (e) {
      setCamError(`Caméra inaccessible : ${e.message}`);
    }
  };

  const stopWebcam = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamRunning(false);
  };

  const drawOverlay = useCallback(() => {
    const cv = overlayRef.current;
    const vd = videoRef.current;
    if (!cv || !vd || vd.videoWidth === 0) return;
    cv.width = vd.videoWidth;
    cv.height = vd.videoHeight;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    overlayBoxes.current.forEach(({ bbox, track_id, sharpness, frameW, frameH }) => {
      const sx = cv.width / (frameW || cv.width);
      const sy = cv.height / (frameH || cv.height);
      const [x1, y1, x2, y2] = bbox;
      const bx = x1 * sx, by = y1 * sy, bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;
      const color = sharpness > 500 ? "#22c55e" : sharpness > 150 ? "#f97316" : "#ef4444";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);
      const label = `#${track_id}`;
      const fs = Math.max(11, Math.min(16, bw * 0.16));
      ctx.font = `bold ${fs}px monospace`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color + "cc";
      ctx.fillRect(bx, by - fs - 5, tw, fs + 5);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + 4, by - 4);
      ctx.fillStyle = color + "66";
      ctx.fillRect(bx, by + bh + 2, bw * Math.min(1, sharpness / 1500), 3);
    });
    rafRef.current = requestAnimationFrame(drawOverlay);
  }, []);

  const startFrameLoop = () => {
    intervalRef.current = setInterval(async () => {
      const vd = videoRef.current;
      const cv = canvasRef.current;
      if (!vd || !cv || vd.readyState < 2 || sending) return;
      cv.width = vd.videoWidth;
      cv.height = vd.videoHeight;
      cv.getContext("2d").drawImage(vd, 0, 0);
      cv.toBlob(async (blob) => {
        if (!blob) return;
        setSending(true);
        try {
          const fd = new FormData();
          fd.append("file", blob, "frame.jpg");
          const r = await fetch(`${API}/detect-frame`, { method: "POST", body: fd });
          if (!r.ok) return;
          const data = await r.json();
          overlayBoxes.current = (data.visible_plates || []).map(p => ({
            bbox: p.bbox, track_id: p.track_id,
            sharpness: p.sharpness, frameW: data.frame_w, frameH: data.frame_h,
          }));
          setLiveVisible(data.visible_plates || []);
          if (data.new_detections?.length > 0) {
            setLiveDetect(prev => [...data.new_detections, ...prev].slice(0, 30));
            if (onDetectionDone) onDetectionDone();
          }
        } catch (_) { }
        finally { setSending(false); }
      }, "image/jpeg", 0.88);
    }, WEBCAM_INTERVAL_MS);
    drawOverlay();
  };

  return (
    <div>
      {/* Mode selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 24,
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 10, padding: 6, width: "fit-content" }}>
        <button onClick={() => { if (camRunning) stopWebcam(); reset(); setMode("file"); }} 
          style={{ background: mode === "file" ? C.accent : "transparent",
            border: "none", borderRadius: 7, color: mode === "file" ? "#fff" : C.textSub,
            padding: "8px 20px", cursor: "pointer", fontSize: "0.82rem", fontWeight: 600 }}>
          🎬 Fichier vidéo
        </button>
        {enableWebcam && <button onClick={() => { reset(); setMode("webcam"); }}
          style={{ background: mode === "webcam" ? C.accent : "transparent",
            border: "none", borderRadius: 7, color: mode === "webcam" ? "#fff" : C.textSub,
            padding: "8px 20px", cursor: "pointer", fontSize: "0.82rem", fontWeight: 600 }}>
          🎥 Webcam live
        </button>}
      </div>

      {/* File mode */}
      {mode === "file" && (
        <div>
          {!file ? (
            <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); loadFile(e.dataTransfer.files[0]); }}
              onClick={() => inputRef.current.click()}
              style={{ border: `2px dashed ${dragging ? C.accent : C.border2}`,
                borderRadius: 14, padding: "60px 24px", textAlign: "center",
                cursor: "pointer", background: dragging ? C.accentBg : C.surface }}>
              <div style={{ width: 64, height: 64, borderRadius: "50%", background: C.accentBg,
                border: `1px solid ${C.border}`, display: "flex", alignItems: "center",
                justifyContent: "center", fontSize: "2rem", margin: "0 auto 18px" }}>🎬</div>
              <p style={{ color: C.text, margin: 0, fontSize: "0.92rem", fontWeight: 600 }}>
                Glissez une vidéo ici ou{" "}
                <span style={{ color: C.accent, textDecoration: "underline" }}>cliquez pour parcourir</span>
              </p>
              <p style={{ color: C.textMute, margin: "8px 0 0", fontSize: "0.72rem" }}>MP4 · AVI · MOV · MKV · WEBM</p>
              <input ref={inputRef} type="file" accept="video/*" style={{ display: "none" }}
                onChange={(e) => loadFile(e.target.files[0])} />
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
              <div>
                <video src={fileURL} controls style={{ width: "100%", borderRadius: 10,
                  border: `1px solid ${C.border}`, background: "#000", maxHeight: 320 }} />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                  <span style={{ color: C.textMute, fontSize: "0.7rem" }}>
                    {file.name} — {(file.size / 1024 / 1024).toFixed(1)} Mo
                  </span>
                  <button onClick={reset} style={{ background: C.surface,
                    border: `1px solid ${C.border2}`, borderRadius: 6, color: C.textSub,
                    padding: "4px 10px", cursor: "pointer", fontSize: "0.7rem" }}>✕ Changer</button>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {!result && !loading && !error && (
                  <button onClick={submitVideo} style={{ background: C.accent, border: "none",
                    borderRadius: 8, color: "#fff", fontWeight: 700, fontSize: "0.88rem",
                    padding: "13px 0", cursor: "pointer", boxShadow: `0 4px 14px ${C.accent}44` }}>
                    Analyser →
                  </button>
                )}
                {loading && <Spinner label={progress || "Traitement…"} />}
                {error && <ErrorBox msg={error} />}
                {result && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={{ background: C.accentBg, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: "14px 16px" }}>
                      <p style={{ margin: 0, fontWeight: 700, color: C.text }}>
                        {result.plates_count} véhicule{result.plates_count > 1 ? "s" : ""} identifié{result.plates_count > 1 ? "s" : ""}
                      </p>
                      <p style={{ margin: "4px 0 0", color: C.textSub, fontSize: "0.72rem" }}>
                        {result.frames_analysed} frames · {result.fps} fps · {result.tracks_total} tracks
                      </p>
                    </div>
                    {result.video_download && (
                      <a href={`${API}${result.video_download}`}
                        download={`platewatch_${result.video_id?.slice(0, 8)}.mp4`}
                        style={{ display: "block", textAlign: "center", background: "#15803d",
                          border: "none", borderRadius: 8, color: "#fff", fontWeight: 700,
                          padding: "11px 0", textDecoration: "none" }}>
                        ⬇ Télécharger la vidéo annotée
                      </a>
                    )}
                    <div style={{ maxHeight: 320, overflowY: "auto", display: "flex", flexDirection: "column", gap: 7 }}>
                      {result.detections?.map((d, i) => <PlateResultCard key={i} item={d} showTrack />)}
                    </div>
                    <button onClick={reset} style={{ background: C.surface,
                      border: `1px solid ${C.border2}`, borderRadius: 8, color: C.textSub,
                      padding: "9px 0", cursor: "pointer", fontSize: "0.76rem", fontWeight: 600 }}>
                      Analyser une autre vidéo
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Webcam mode */}
      {mode === "webcam" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 22 }}>
          <div>
            <div style={{ position: "relative", borderRadius: 12, overflow: "hidden",
              background: "#000", border: `1px solid ${C.border}` }}>
              <video ref={videoRef} muted playsInline style={{ width: "100%", display: "block", maxHeight: 420 }} />
              <canvas ref={overlayRef} style={{ position: "absolute", top: 0, left: 0,
                width: "100%", height: "100%", pointerEvents: "none" }} />
              <canvas ref={canvasRef} style={{ display: "none" }} />
              {camRunning && (
                <div style={{ position: "absolute", top: 10, left: 10, display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ width: 9, height: 9, borderRadius: "50%", background: "#ef4444",
                    display: "inline-block", animation: "pulse 1s infinite" }} />
                  <span style={{ color: "#fff", fontSize: "0.7rem", fontWeight: 700 }}>EN DIRECT</span>
                </div>
              )}
            </div>
            <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
              {!camRunning ? (
                <button onClick={startWebcam} style={{ flex: 1, background: "#15803d",
                  border: "none", borderRadius: 8, color: "#fff", fontWeight: 700,
                  padding: "11px 0", cursor: "pointer" }}>▶ Démarrer la caméra</button>
              ) : (
                <button onClick={stopWebcam} style={{ flex: 1, background: "#dc2626",
                  border: "none", borderRadius: 8, color: "#fff", fontWeight: 700,
                  padding: "11px 0", cursor: "pointer" }}>⏹ Arrêter</button>
              )}
            </div>
            {camError && <ErrorBox msg={camError} />}
          </div>
          <div>
            <SectionHeader count={liveDetect.length}>Véhicules identifiés</SectionHeader>
            <div style={{ maxHeight: 460, overflowY: "auto", display: "flex", flexDirection: "column", gap: 7 }}>
              {liveDetect.length === 0 ? (
                <div style={{ background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: "32px 16px", textAlign: "center" }}>
                  <span style={{ fontSize: "2rem", display: "block", marginBottom: 8 }}>🔀</span>
                  <p style={{ color: C.textMute, fontSize: "0.82rem", margin: 0 }}>
                    {camRunning ? "Surveillance active…" : "Démarrez la caméra"}
                  </p>
                </div>
              ) : (
                liveDetect.map((item, i) => <PlateResultCard key={i} item={item} showTrack />)
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Petits composants UI partagés ─────────────────────────────────────────────

function Spinner({ label, sub }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: "28px 20px", textAlign: "center" }}>
      <div style={{ width: 34, height: 34, border: `3px solid ${C.border}`,
        borderTop: `3px solid ${C.accent}`, borderRadius: "50%",
        animation: "spin 0.9s linear infinite", margin: "0 auto 12px" }} />
      <p style={{ color: C.text, fontSize: "0.82rem", margin: 0, fontWeight: 600 }}>{label}</p>
      {sub && <p style={{ color: C.textMute, fontSize: "0.68rem", margin: "5px 0 0" }}>{sub}</p>}
    </div>
  );
}

function ErrorBox({ msg }) {
  return (
    <div style={{ background: "#fef2f2", border: "1px solid #fecaca",
      borderRadius: 10, padding: "13px 16px" }}>
      <p style={{ color: "#b91c1c", fontWeight: 700, margin: 0, fontSize: "0.8rem" }}>Erreur</p>
      <p style={{ color: "#dc2626", fontSize: "0.75rem", margin: "3px 0 0" }}>{msg}</p>
    </div>
  );
}

function AlertTag() {
  return (
    <span style={{ background: "#fef2f2", color: "#b91c1c",
      border: "1px solid #fecaca", borderRadius: 5,
      padding: "3px 9px", fontSize: "0.65rem", fontWeight: 700 }}>
      ⚠ ALERTE CRÉÉE
    </span>
  );
}

function ConfScore({ conf }) {
  return (
    <div>
      <span style={{ color: C.textMute, fontSize: "0.63rem", display: "block", marginBottom: 2 }}>CONFIANCE</span>
      <span style={{ color: conf > 0.8 ? "#15803d" : conf > 0.6 ? "#c2410c" : "#b91c1c",
        fontWeight: 800, fontSize: "1.1rem" }}>
        {(conf * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function DBStatus({ found }) {
  return (
    <div>
      <span style={{ color: C.textMute, fontSize: "0.63rem", display: "block", marginBottom: 2 }}>VÉHICULE</span>
      <span style={{ color: found ? "#15803d" : "#6d28d9", fontSize: "0.82rem", fontWeight: 700 }}>
        {found ? "✓ Trouvé en DB" : "Non répertorié"}
      </span>
    </div>
  );
}

function Dashboard({ detections, alerts }) {
  const now = Date.now();
  const last24h     = detections.filter(d => now - parseDate(d.detected_at) < 86400000);
  const alertsToday = alerts.filter(a => now - parseDate(a.temps) < 86400000);
  const voles       = alerts.filter(a => a.infraction === "vole").length;
  const byHour = Array(24).fill(0);
  last24h.forEach(d => { byHour[parseDate(d.detected_at).getHours()]++; });
  const maxH = Math.max(...byHour, 1);

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 28 }}>
        <StatCard label="Détections 24h"   value={last24h.length}     accent="#2563eb" />
        <StatCard label="Alertes 24h"      value={alertsToday.length} accent="#ea580c" />
        <StatCard label="Véhicules volés"  value={voles}              accent="#dc2626" />
        <StatCard label="Total détections" value={detections.length}  accent="#059669" sub="depuis le début" />
      </div>
      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12,
        padding: "20px 24px", marginBottom: 28, boxShadow: "0 1px 6px #0000000a" }}>
        <SectionHeader>Activité par heure (aujourd'hui)</SectionHeader>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 72 }}>
          {byHour.map((v, h) => {
            const isNow = new Date().getHours() === h;
            return (
              <div key={h} title={`${h}h : ${v}`} style={{ flex: 1,
                background: isNow ? "#2563eb" : v > 0 ? "#93c5fd" : C.border,
                height: `${Math.round((v / maxH) * 100)}%`, minHeight: 4,
                borderRadius: "3px 3px 0 0", transition: "height .3s" }} />
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between",
          color: C.textMute, fontSize: "0.6rem", marginTop: 6 }}>
          <span>0h</span><span>6h</span><span>12h</span><span>18h</span><span>23h</span>
        </div>
      </div>
      <SectionHeader count={alertsToday.length}>Alertes récentes</SectionHeader>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {alerts.slice(0, 6).map(a => {
          const s = STATUS[a.infraction] || STATUS.inconnu;
          return (
            <div key={a.id} style={{ background: C.surface, border: `1px solid ${C.border}`,
              borderLeft: `4px solid ${s.dot}`, borderRadius: 8, padding: "12px 16px",
              display: "flex", alignItems: "center", gap: 14, boxShadow: "0 1px 4px #0000000a" }}>
              <StatusBadge statut={a.infraction} size="lg" />
              <Plate text={a.immatriculation} />
              <span style={{ color: C.textMute, fontSize: "0.72rem", marginLeft: "auto" }}>{fmt(a.temps)}</span>
            </div>
          );
        })}
        {alerts.length === 0 && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "24px", textAlign: "center", color: C.textMute }}>✓ Aucune alerte</div>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  DETECTIONS / ALERTS / VEHICULES
// ════════════════════════════════════════════════════════════════════════════

function DetectionsView({ detections, loading }) {
  const [filter, setFilter] = useState("");
  const filtered = detections.filter(d => d.immatriculation.toLowerCase().includes(filter.toLowerCase()));
  return (
    <div>
      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        <input value={filter} onChange={e => setFilter(e.target.value)}
          placeholder="Filtrer par immatriculation…"
          style={{ background: C.surface, border: `1px solid ${C.border2}`, borderRadius: 7,
            color: C.text, padding: "8px 14px", fontSize: "0.82rem", outline: "none", width: 280 }} />
        <span style={{ color: C.textMute, fontSize: "0.78rem", marginLeft: "auto", alignSelf: "center" }}>
          {loading ? "↻ chargement…" : `${filtered.length} résultats`}
        </span>
      </div>
      <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
        overflow: "hidden", boxShadow: "0 1px 6px #0000000a" }}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
            <thead>
              <tr style={{ background: C.surface2, borderBottom: `1px solid ${C.border}` }}>
                {["ID","Immatriculation","Confiance","Source","Caméra","Date"].map(h => (
                  <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 700,
                    color: C.textSub, textTransform: "uppercase", fontSize: "0.64rem" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((d, i) => (
                <tr key={d.id} style={{ background: i%2===0 ? C.surface : C.surface2,
                  borderBottom: `1px solid ${C.border}` }}>
                  <td style={{ padding: "10px 14px", color: C.textMute }}>#{d.id}</td>
                  <td style={{ padding: "10px 14px" }}><Plate text={d.immatriculation} /></td>
                  <td style={{ padding: "10px 14px" }}>
                    <span style={{ color: d.confiance>0.8?"#15803d":d.confiance>0.6?"#c2410c":"#b91c1c", fontWeight:700 }}>
                      {(d.confiance*100).toFixed(1)}%
                    </span>
                  </td>
                  <td style={{ padding: "10px 14px", color: C.textSub }}>{d.source_type}</td>
                  <td style={{ padding: "10px 14px", color: C.textSub }}>{d.camera_id ?? "—"}</td>
                  <td style={{ padding: "10px 14px", color: C.textMute }}>{fmt(d.detected_at)}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={6} style={{ padding: 28, color: C.textMute, textAlign: "center" }}>Aucune détection.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function AlertsView({ alerts }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {alerts.map(a => {
        const s = STATUS[a.infraction] || STATUS.inconnu;
        return (
          <div key={a.id} style={{ background: s.bg, border: `1px solid ${s.border}`,
            borderLeft: `5px solid ${s.dot}`, borderRadius: 9, padding: "14px 18px",
            display: "flex", alignItems: "center", gap: 18, boxShadow: "0 1px 5px #0000000a" }}>
            <span style={{ color: C.textMute, fontSize: "0.68rem", minWidth: 28 }}>#{a.id}</span>
            <StatusBadge statut={a.infraction} size="lg" />
            <Plate text={a.immatriculation} />
            <span style={{ color: C.textSub, fontSize: "0.73rem", marginLeft: "auto" }}>{fmt(a.temps)}</span>
          </div>
        );
      })}
      {alerts.length === 0 && (
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
          padding: 36, textAlign: "center", color: C.textMute }}>✓ Aucune alerte</div>
      )}
    </div>
  );
}

function VehiclesView({ vehicles }) {
  const [filter, setFilter]       = useState("");
  const [filterStatut, setFilterS] = useState("tous");
  const filtered = vehicles.filter(v =>
    v.immatriculation.toLowerCase().includes(filter.toLowerCase()) &&
    (filterStatut === "tous" || v.statut === filterStatut)
  );
  return (
    <div>
      <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        <input value={filter} onChange={e => setFilter(e.target.value)}
          placeholder="Rechercher une immatriculation…"
          style={{ background: C.surface, border: `1px solid ${C.border2}`, borderRadius: 7,
            color: C.text, padding: "8px 14px", fontSize: "0.82rem", outline: "none", width: 260 }} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {["tous","safe","vole","suspicieux","inconnu"].map(s => {
            const isActive = filterStatut === s;
            const col = s==="tous"?C.accent:s==="safe"?"#15803d":s==="vole"?"#b91c1c":s==="suspicieux"?"#c2410c":"#6d28d9";
            return (
              <button key={s} onClick={() => setFilterS(s)} style={{ background: isActive ? col : C.surface,
                border: `1px solid ${isActive ? col : C.border2}`, borderRadius: 6,
                color: isActive ? "#fff" : C.textSub, padding: "6px 12px",
                cursor: "pointer", fontSize: "0.72rem", fontWeight: 600 }}>
                {s.toUpperCase()}
              </button>
            );
          })}
        </div>
        <span style={{ color: C.textMute, fontSize: "0.78rem", alignSelf: "center", marginLeft: "auto" }}>
          {filtered.length} véhicule{filtered.length>1?"s":""}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(250px,1fr))", gap: 12 }}>
        {filtered.map(v => {
          const s = STATUS[v.statut] || STATUS.inconnu;
          return (
            <div key={v.id} style={{ background: C.surface, border: `1px solid ${C.border}`,
              borderTop: `3px solid ${s.dot}`, borderRadius: 9, padding: "14px 16px",
              display: "flex", flexDirection: "column", gap: 8, boxShadow: "0 1px 5px #0000000a" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <Plate text={v.immatriculation} />
                <StatusBadge statut={v.statut} />
              </div>
              <span style={{ color: C.textMute, fontSize: "0.65rem" }}>ID #{v.id}</span>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div style={{ color: C.textMute, gridColumn: "1/-1", textAlign: "center", padding: 36,
            background: C.surface, borderRadius: 10, border: `1px solid ${C.border}` }}>Aucun véhicule.</div>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════════════
//  APP RACINE
// ════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [tab,        setTab]    = useState("upload");   // la démo commence par l'analyse d'image
  const [detections, setDetect] = useState([]);
  const [alerts,     setAlerts] = useState([]);
  const [vehicles,   setVeh]    = useState([]);
  const [loading,    setLoad]   = useState(true);
  const [apiOk,      setApiOk]  = useState(null);
  const [lastSync,   setSync]   = useState(null);
  const polling = useRef(null);
  const [cfg, setCfg] = useState({ demo_mode: false, persist_detections: true,
    enable_video: true, enable_webcam: true, max_image_mb: 8 });
  const persist = cfg.persist_detections;

  useEffect(() => {
    apiFetch("/config").then(setCfg).catch(() => {});
    if (STATIC) {   // télécharge les modèles pendant que le visiteur regarde les exemples
      import("./lib/localDetector.js").then(m => m.warmup()).then(() => { _ready = true; }).catch(() => {});
    }
  }, []);

  // Démo publique : rien n'est stocké sur le serveur, l'historique vit dans le navigateur.
  const onImageDone = useCallback((data) => {
    if (persist || !data?.detection) return;
    const id = Date.now();
    setDetect(prev => [{ id, ...data.detection }, ...prev]);
    if (data.alert) setAlerts(prev => [{ id, ...data.alert }, ...prev]);
  }, [persist]);

  const fetchAll = useCallback(async () => {
    try {
      if (!persist) {
        setVeh(await apiFetch("/vehicules"));
      } else {
      const [d, a, v] = await Promise.all([
        apiFetch("/detections"), apiFetch("/alerts"), apiFetch("/vehicules"),
      ]);
      setDetect([...d].sort((x,y) => parseDate(y.detected_at) - parseDate(x.detected_at)));
      setAlerts([...a].sort((x,y) => parseDate(y.temps) - parseDate(x.temps)));
      setVeh(v);
      }
      setApiOk(true);
      setSync(new Date());
    } catch { setApiOk(false); }
    finally  { setLoad(false); }
  }, [persist]);

  useEffect(() => {
    fetchAll();
    if (!persist) return;                       // pas de polling inutile en démo
    polling.current = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(polling.current);
  }, [fetchAll, persist]);

  const newAlerts = alerts.filter(a => Date.now() - parseDate(a.temps) < 86400000).length;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "system-ui,'Segoe UI',sans-serif" }}>

      <header style={{ background: C.header, borderBottom: "3px solid #c8a94a",
        padding: "0 32px", display: "flex", alignItems: "center",
        height: 60, gap: 16, boxShadow: "0 2px 12px #00000022" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg,#c8a94a,#f0d080)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "1.2rem", boxShadow: "0 2px 6px #c8a94a66" }}>🚗</div>
          <div>
            <div style={{ fontSize: "0.9rem", fontWeight: 800, color: "#fff" }}>
              Système de détection de fraude d'immatriculation
            </div>
            <div style={{ fontSize: "0.6rem", color: "#93c5fd", letterSpacing: "0.08em" }}>
              Surveillance — المملكة المغربية
            </div>
          </div>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8,
            background: "rgba(255,255,255,0.08)", borderRadius: 7, padding: "5px 12px",
            border: "1px solid rgba(255,255,255,0.12)" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%",
              background: apiOk===null?"#94a3b8":apiOk?"#22c55e":"#ef4444",
              boxShadow: apiOk?"0 0 6px #22c55e":undefined }} />
            <span style={{ fontSize: "0.67rem", color: apiOk?"#86efac":"#fca5a5" }}>
              {apiOk===null?"Connexion…":apiOk?(cfg.static?"Analyse locale":"API connectée"):"API hors ligne"}
            </span>
          </div>
          {lastSync && (
            <span style={{ fontSize: "0.62rem", color: "#93c5fd", opacity: 0.7 }}>
              Sync {lastSync.toLocaleTimeString("fr-MA")}
            </span>
          )}
          <button onClick={fetchAll} style={{ background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.2)", borderRadius: 7, color: "#cbd5e1",
            padding: "5px 12px", cursor: "pointer", fontSize: "0.72rem", fontWeight: 600 }}>
            ↻ Actualiser
          </button>
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 24px" }}>
        {cfg.demo_mode && (
          <div style={{ background: "#fffbeb", border: "1px solid #fde68a", color: "#92400e",
            borderRadius: 10, padding: "10px 16px", fontSize: "0.74rem", marginBottom: 16 }}>
            <b>Démo publique.</b> {cfg.static
              ? "L'analyse s'exécute dans votre navigateur : votre image n'est jamais envoyée à un serveur."
              : "Vos images sont analysées en mémoire et ne sont pas conservées."}{" "}
            Le modèle est entraîné sur des plaques marocaines ; la base de véhicules est fictive.
          </div>
        )}
        <TabBar active={tab} onChange={setTab} alertCount={newAlerts}
          tabs={TABS.filter(t => t.id !== "video" || cfg.enable_video)} />
        {tab === "dashboard"  && <Dashboard   detections={detections} alerts={alerts} />}
        {tab === "upload" && (
  <ImageView onDetectionDone={(d) => { onImageDone(d); fetchAll(); }} maxMb={cfg.max_image_mb} />
)}

{tab === "video" && (
  <VideoView onDetectionDone={fetchAll} enableWebcam={cfg.enable_webcam} />
)}
        
        {tab === "detections" && <DetectionsView detections={detections} loading={loading} />}
        {tab === "alerts"     && <AlertsView  alerts={alerts} />}
        {tab === "vehicules"  && <VehiclesView vehicles={vehicles} />}
      </main>

      <style>{`
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.3 } }
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border2}; border-radius: 3px; }
        button:hover { opacity: 0.88; }
        input:focus { border-color: ${C.accent} !important; box-shadow: 0 0 0 3px ${C.accent}22 !important; }
      `}</style>
    </div>
  );
}