# -*- coding: utf-8 -*-
"""
model_02.py
Architecture CRNN (CNN + BiLSTM + CTC) pour plaques marocaines.


 VUE D'ENSEMBLE :

  Image plaque (32x128 px, niveaux de gris)
        |
    [CNN ResNet]
        |  Extrait les features visuelles : courbes des chiffres,
        |  forme des lettres. Sortie : (8, 32, 64)
        |  = 8 lignes x 32 colonnes x 64 descripteurs
        |
    [Reshape + Permute]
        |  Transforme la carte 2D en sequence 1D.
        |  Chaque "colonne" de la plaque devient un pas de temps.
        |  (8, 32, 64) -> (32, 512)
        |  32 pas de temps, chacun decrit par 512 valeurs
        |
    [BiLSTM]
        |  Lit la sequence dans les 2 sens (gauche->droite ET droite->gauche).
        |  Comprend le contexte : le "1" dans "91620" est different du "1" seul
        |  car le BiLSTM voit ce qui l'entoure.
        |  Sortie : (32, 128) = 32 pas x 128 valeurs de contexte
        |
    [Dense + Softmax]
        |  Pour chaque pas de temps, predit la probabilite
        |  de chaque caractere du vocabulaire.
        |  Sortie : (32, NUM_CLASSES)
        |
    [CTC Loss pendant l'entrainement]
        |  Aligne automatiquement les predictions (32 pas)
        |  avec le label court (7-8 chars) sans segmentation.
        |
    [Decodage CTC greedy a l'inference]  maintenant c'est beam_search_decode
           Collapse les repetitions et supprime les blanks.
           "---9-90-0-1-20--ھ--7-72-" -> "90120ھ72"

VOCABULAIRE :

  Index 0       : blank CTC (token special, jamais dans le texte final)
  Index 1..10   : chiffres 0-9
  Index 11..N   : lettres arabes
  Total         : NUM_CLASSES = len(VOCAB) + 1
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers





VOCAB = (
    "0123456789"
    "أبھدو"
)
# 1-base : l'index 0 est RESERVE pour le blank CTC maintenant c'est le dernier avec le beam_search_decode
# Exemple : '0' -> 1, '1' -> 2, ..., 'أ' -> 11, ...
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)} # a chaque caractere on associe un index entier
 
# Inverse : index -> caractere (pour le decodage)
IDX_TO_CHAR = {i : c for i, c in enumerate(VOCAB)} # a chaque index on associe un caractere
NUM_CLASSES = len(VOCAB) + 1  # +1 pour le blank CTC a l'index NUM_CLASSES -1

BLANK_IDX = NUM_CLASSES - 1

IDX_TO_CHAR[BLANK_IDX] = ""

# Dimensions de l'image d'entree
IMG_HEIGHT = 32
IMG_WIDTH  = 128




# NORMALISATION DES LABELS

# Certaines annotations peuvent contenir ه (U+0647) par erreur de frappe.
# On les normalise vers ھ (U+06BE) qui est le caractere de notre VOCAB.
_NORMALIZE_MAP = {
    "\u0647": "\u06BE",  # ه -> ھ  (heh -> heh doachashmee)
    "\u0629": "\u06BE",  # ة -> ھ  (teh marbuta si erreur de saisie)
}

def normalize_label(text): 
    """
    Normalise les variantes de caracteres vers leur forme dans VOCAB.

    Pourquoi ?
      Sur un clavier arabe, il est facile de taper ه au lieu de ھ.
      Cette fonction corrige silencieusement ces erreurs de saisie
      pour que label_to_indices ne rencontre pas de caracteres inconnus.

    Exemple :
      "56007ه1"  ->  "56007ھ1"   (correction automatique)
      "90120أ72" ->  "90120أ72"  (inchange, deja correct)
    """
    return "".join(_NORMALIZE_MAP.get(c, c) for c in text)


# BLOC RESIDUEL 

def residual_block(x, filters, strides=1, dropout=0.2):
    """
    Bloc residuel : sortie = F(x) + x

    Pourquoi des blocs residuels ?
      Sans connexion de saut, les gradients s'evaporent dans les reseaux
      profonds (le "vanishing gradient problem"). La connexion de saut
      permet au gradient de "court-circuiter" les couches et de remonter
      efficacement jusqu'aux premieres couches.

    Schema :
      x --> [Conv 3x3] -> [BN] -> [LeakyReLU]
              -> [Conv 3x3] -> [BN] --> (+) -> [LeakyReLU] -> [Dropout]
      x --> [Conv 1x1 si dimensions differentes] --------^

    Parametres :
      filters  : nombre de filtres de convolution (= dimension de sortie)
      strides  : 1 = taille conservee, 2 = resolution divisee par 2
      dropout  : taux de regularisation (evite le surapprentissage)
    """
    # -- Chemin principal : deux convolutions --
    out = layers.Conv2D(filters, 3, strides=strides,
                        padding="same", use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)

    out = layers.Conv2D(filters, 3, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)

    # -- Connexion de saut --
    # Si les dimensions different (stride!=1 ou nb_filtres change),
    # on ajuste x avec une convolution 1x1 sans activation.
    if strides != 1 or x.shape[-1] != filters:
        x = layers.Conv2D(filters, 1, strides=strides, use_bias=False)(x)

    # -- Addition et activation finale --
    out = layers.Add()([out, x])
    out = layers.LeakyReLU()(out)
    out = layers.Dropout(dropout)(out)
    return out



# MODELE CRNN COMPLET

def build_crnn_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                     num_classes=NUM_CLASSES, dropout=0.2):
    """
    Construit le modele CRNN.

    Dimensions etape par etape (B = batch_size) :
      Entree    : (B, 32, 128, 1)    image gris normalisee
      Bloc 1    : (B, 32, 128, 16)   features legeres, meme taille
      Bloc 2    : (B, 16,  64, 16)   resolution /2
      Bloc 2b   : (B, 16,  64, 16)   affinage
      Bloc 3    : (B,  8,  32, 32)   resolution /2
      Bloc 3b   : (B,  8,  32, 32)   affinage
      Bloc 4    : (B,  8,  32, 64)   features riches, meme taille
      Bloc 4b   : (B,  8,  32, 64)   affinage
      Permute   : (B, 32,   8, 64)   echanger largeur et hauteur
      Reshape   : (B, 32,     512)   aplatir -> sequence temporelle
      BiLSTM    : (B, 32,     128)   contexte bidirectionnel
      Dense     : (B, 32, N_cls)     probabilites par pas de temps
    """

    # ----- ENTREE -----
    inputs = layers.Input(shape=(img_height, img_width, 1), name="image")

    # Normalisation pixels 0-255 -> 0-1
    # Rescaling est une couche Keras standard, serialisable sans probleme
    x = layers.Rescaling(scale=1.0 / 255.0)(inputs)

    # ----- CNN : EXTRACTION DES FEATURES -----
    x = residual_block(x, 16, strides=1, dropout=dropout)   # (32,128,16)

    x = residual_block(x, 16, strides=2, dropout=dropout)   # (16, 64,16)
    x = residual_block(x, 16, strides=1, dropout=dropout)   # affinage

    x = residual_block(x, 32, strides=2, dropout=dropout)   # ( 8, 32,32)
    x = residual_block(x, 32, strides=1, dropout=dropout)   # affinage

    x = residual_block(x, 64, strides=1, dropout=dropout)   # ( 8, 32,64)
    x = residual_block(x, 64, strides=1, dropout=dropout)   # affinage

    # ----- TRANSITION CNN -> LSTM -----
    # On veut une sequence ou chaque element = une "colonne" de la plaque.
    # Actuellement x = (B, 8, 32, 64) : 8 lignes, 32 colonnes, 64 features.
    # On echange lignes et colonnes -> (B, 32, 8, 64)
    x = layers.Permute((2, 1, 3))(x)

    # On aplatit les 8 lignes et 64 features en un seul vecteur de 512
    # -> (B, 32, 512) : 32 pas de temps, chacun decrit par 512 valeurs
    time_steps = x.shape[1]               # 32
    features   = x.shape[2] * x.shape[3]  # 8 * 64 = 512
    x = layers.Reshape((time_steps, features))(x)

    # ----- BiLSTM -----
    # return_sequences=True : on veut une prediction a CHAQUE pas de temps
    # (pas juste a la fin de la sequence)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=dropout)
    )(x)
    # Sortie : (B, 32, 128)  <- 128 = 64 (avant) + 64 (arriere)

    # ----- SORTIE -----
    # Probabilite de chaque classe a chaque pas de temps
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)
    # Sortie : (B, 32, num_classes)

    return keras.Model(inputs=inputs, outputs=output,
                       name="CRNN_Plaque_Marocaine")


# CTC LOSS

class CTCLoss(keras.losses.Loss):
    """
    CTC Loss avec blank_index=0 EXPLICITE.

    POURQUOI NE PAS UTILISER ctc_batch_cost DE KERAS ?
    ===================================================
    keras.backend.ctc_batch_cost utilise en interne tf.nn.ctc_loss
    avec blank = num_classes - 1 (dernier index). C'est un choix fixe
    qu'on ne peut pas changer.

    Notre VOCAB commence a l'index 1 (0 est reserve au blank).
    Si on utilisait ctc_batch_cost, TensorFlow verrait le blank au
    dernier index, qui correspond a un VRAI caractere dans notre VOCAB.
    -> Conflit -> erreur "non-null label following null label"

    SOLUTION :
    Utiliser tf.nn.ctc_loss directement avec blank_index=0.
    Cela correspond exactement a notre convention : index 0 = blank,
    index 1..N = vrais caracteres.

    NOTE SUR **kwargs :
    Keras passe automatiquement des arguments comme 'reduction' lors
    de la sauvegarde/chargement du modele. **kwargs les absorbe
    et les transmet a la classe parente sans erreur.
    """

    def __init__(self, name="CTCLoss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """
        Calcule la perte CTC.

        Arguments :
          y_true : (batch, MAX_LABEL_LEN) int32
                   indices des caracteres, -1 pour le padding
          y_pred : (batch, time_steps, num_classes) float32
                   probabilites softmax du modele

        Retourne : scalaire float, la perte moyenne du batch
        """
        batch_size   = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]  # 32 pas de temps

        # Longueur des sequences de predictions (32 pour toutes)
        input_lengths = tf.fill([batch_size], tf.cast(input_length, tf.int32))

        # Longueur reelle de chaque label (nombre de chars != -1)
        label_lengths = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1
        )

        # Remplacer le padding (-1) par 0 pour eviter des indices negatifs
        # Ces positions seront ignorees grace a label_lengths
        y_true_clean = tf.cast(
            tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true),
            tf.int32
        )

        # tf.nn.ctc_loss attend les logits en format (time, batch, classes)
        # -> on transpose depuis (batch, time, classes)
        # On prend le log des probabilites (logits pour CTC)
        logits_transposed = tf.transpose(
            tf.math.log(y_pred + 1e-8),
            perm=[1, 0, 2]
        )

        loss = tf.nn.ctc_loss(
            labels=y_true_clean,
            logits=logits_transposed,
            label_length=label_lengths,
            logit_length=input_lengths,
            blank_index=BLANK_IDX,       # 0 : coherent avec notre VOCAB
            logits_time_major=True,
        )
        return tf.reduce_mean(loss)


# ------------------------------------------------------------------
# DECODAGE CTC
# ------------------------------------------------------------------

# def decode_predictions(y_pred, idx_to_char=IDX_TO_CHAR, blank_idx=BLANK_IDX):
#     """
#     Decodage CTC greedy MANUEL, coherent avec blank_index=0.

#     POURQUOI PAS ctc_decode DE KERAS ?
#     ====================================
#     keras.backend.ctc_decode suppose TOUJOURS blank = dernier index.
#     Or on entraine avec blank = index 0.
#     -> Utiliser ctc_decode donnerait des resultats incorrects.
#     -> On implemente le decodage greedy a la main, c'est simple.

#     Algorithme greedy (best path decoding) :
#       1. A chaque pas de temps, prendre l'indice avec la probabilite max
#          -> sequence brute, ex: [0,9,9,9,1,1,2,0,1,0,0,11,0,8,8,3,0,0]
#       2. Supprimer les repetitions consecutives
#          -> [0,9,1,2,0,1,0,11,0,8,3,0]
#       3. Supprimer les blanks (index 0)
#          -> [9,1,2,1,11,8,3]
#       4. Convertir en texte via IDX_TO_CHAR
#          -> "90120ھ72"  (si 11 = 'ھ', etc.)

#     Parametres :
#       y_pred    : (batch, time_steps, num_classes) float32
#       idx_to_char : dictionnaire index -> caractere
#       blank_idx : index du token blank (0 dans notre cas)

#     Retourne : liste de chaines de caracteres
#     """
#     # Etape 1 : indice le plus probable a chaque pas de temps
#     best_path = np.argmax(y_pred, axis=-1)  # (batch, time_steps)

#     results = []
#     for path in best_path:
#         # Etape 2 : supprimer les repetitions consecutives
#         collapsed = [int(path[0])]
#         for idx in path[1:]:
#             if int(idx) != collapsed[-1]:
#                 collapsed.append(int(idx))

#         # Etapes 3 et 4 : supprimer blanks et convertir
#         text = "".join(
#             idx_to_char.get(idx, "")
#             for idx in collapsed
#             if idx != blank_idx
#         )
#         results.append(text)

#     return results


def decode_predictions(
    y_pred,
    beam_width=10
):
    """
    Decodage CTC Beam Search standard.
    Compatible avec blank = dernier index.
    """

    input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]

    decoded, _ = keras.backend.ctc_decode(
        y_pred,
        input_length=input_length,
        greedy=False,
        beam_width=beam_width
    )

    decoded = decoded[0].numpy()

    results = []

    for seq in decoded:

        text = ""

        for idx in seq:

            if idx == -1:
                continue

            if idx == BLANK_IDX:
                continue

            text += IDX_TO_CHAR.get(int(idx), "")

        results.append(text)

    return results

# ------------------------------------------------------------------
# TEST RAPIDE
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Vocabulaire ({len(VOCAB)} classes + 1 blank = {NUM_CLASSES} total)")
    print(f"VOCAB = '{VOCAB}'")
    print(f"Blank CTC : index {BLANK_IDX}")
    print(f"Premier char ('{VOCAB[0]}') : index {CHAR_TO_IDX[VOCAB[0]]}")
    print(f"Dernier char ('{VOCAB[-1]}') : index {CHAR_TO_IDX[VOCAB[-1]]}")
    print()

    # Test de normalisation
    test = "56007ه1"
    print(f"Normalisation : '{test}' -> '{normalize_label(test)}'")
    print()

    model = build_crnn_model()
    model.summary()
    print(f"\nParametres : {model.count_params():,}")
    print(f"Entree     : {model.input_shape}")
    print(f"Sortie     : {model.output_shape}")