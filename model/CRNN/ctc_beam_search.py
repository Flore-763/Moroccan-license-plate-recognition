"""
ctc_beam_search.py
==================
Décodeur Beam Search CTC pur NumPy.
Compatible avec toute sortie (batch, T, C) d'un modèle Keras.
"""

import numpy as np
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────

def _log_sum_exp(a, b):
    NEG = float('-inf')
    if a == NEG: return b
    if b == NEG: return a
    if a > b: return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ctc_beam_search_decode(probs, beam_width=10, blank_idx=None):
    """
    Décode une séquence (T, C) avec Beam Search CTC.

    probs      : np.ndarray (T, C) — probabilités softmax ou logits
    beam_width : nombre de faisceaux
    blank_idx  : index du blank (défaut : C-1)

    Retourne (indices: list[int], score: float)
    """
    T, C = probs.shape
    if blank_idx is None:
        blank_idx = C - 1

    # Si logits → softmax → log
    if probs.max() > 1.0 + 1e-5:
        probs = _softmax(probs)
    log_p = np.log(np.clip(probs, 1e-10, 1.0))

    NEG = float('-inf')
    # beams : { prefix_tuple : (p_blank_log, p_non_blank_log) }
    beams = {(): (0.0, NEG)}

    for t in range(T):
        new_beams = defaultdict(lambda: (NEG, NEG))

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_sum_exp(p_b, p_nb)

            for c in range(C):
                lp = log_p[t, c]

                if c == blank_idx:
                    nb, nnb = new_beams[prefix]
                    new_beams[prefix] = (_log_sum_exp(nb, p_total + lp), nnb)
                else:
                    ext = prefix + (c,)
                    nb, nnb = new_beams[ext]
                    if prefix and prefix[-1] == c:
                        new_beams[ext] = (nb, _log_sum_exp(nnb, p_b + lp))
                    else:
                        new_beams[ext] = (nb, _log_sum_exp(nnb, p_total + lp))

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: _log_sum_exp(x[1][0], x[1][1]),
                   reverse=True)[:beam_width]
        )

    best = max(beams, key=lambda p: _log_sum_exp(beams[p][0], beams[p][1]))
    return list(best), _log_sum_exp(beams[best][0], beams[best][1])


def _greedy_collapse(seq, blank_idx):
    out, prev = [], None
    for idx in seq:
        if idx != prev and idx != blank_idx:
            out.append(int(idx))
        prev = idx
    return out


def _indices_to_text(indices, characters):
    chars = list(characters)
    return ''.join(chars[i] for i in indices if 0 <= i < len(chars))


def batch_beam_search(y_pred, characters, beam_width=10, blank_idx=None):
    """
    Décode un batch (N, T, C).
    Retourne (list[str], list[float]).
    """
    C = y_pred.shape[-1]
    if blank_idx is None:
        blank_idx = C - 1
    texts, scores = [], []
    for i in range(y_pred.shape[0]):
        probs = y_pred[i]
        if probs.max() > 1.0 + 1e-5:
            probs = _softmax(probs)
        seq, score = ctc_beam_search_decode(probs, beam_width, blank_idx)
        texts.append(_indices_to_text(seq, characters))
        scores.append(score)
    return texts, scores


def batch_greedy_decode(y_pred, characters, blank_idx=None):
    C = y_pred.shape[-1]
    if blank_idx is None:
        blank_idx = C - 1
    texts = []
    for i in range(y_pred.shape[0]):
        seq = np.argmax(y_pred[i], axis=-1)
        texts.append(_indices_to_text(_greedy_collapse(seq, blank_idx), characters))
    return texts