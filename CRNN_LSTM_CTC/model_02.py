# -*- coding: utf-8 -*-
"""
model_02.py
===========
Architecture CRNN (CNN + BiLSTM + CTC) pour plaques marocaines.

COMMENT CA MARCHE :
===================

  Image plaque (32x128)
        |
        |                    Sortie : (8, 32, 64) = 8 lignes x 32 colonnes x 64 features
        |
    [Reshape]             <- Transforme en sequence temporelle
        |                    (8, 32, 64) -> (256, 64)
        |                    256 "pas de temps", chacun avec 64 features
        |
    [BiLSTM]              <- Lit la sequence dans les 2 sens
        |                    Comprend le contexte (un "9" apres un "1" est different)
        |
    [Dense + Softmax]     <- Predit, pour chaque pas de temps, la probabilite
        |                    de chaque caractere de notre vocabulaire
        |
    [Decodage CTC]        <- Transforme les probabilites en texte final
                             "---9-90-0-1-11-2-20--ا--7-72-2-" -> "90120ا72"

VOCABULAIRE (39 classes + 1 blank CTC = 40) :
  Chiffres  : 0 1 2 3 4 5 6 7 8 9
  Arabes    : ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي + variantes
  Blank CTC : token special utilise uniquement pendant l'entrainement

Necessite :
  pip install tensorflow  (>= 2.10)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ------------------------------------------------------------------
# Vocabulaire
# ------------------------------------------------------------------

# Tous les caracteres possibles sur une plaque marocaine
# On n'inclut PAS les barres | car le modele les ignore naturellement

""" VOCAB = (
    "0123456789"
    "أابتثجحخدذرزسشصضطظعغفقكلمنهويىإآؤئءھ"
) """
# APRES : ه remplace par ھ
VOCAB = (
    "0123456789"
    "أابتثجحخدذرزسشصضطظعغفقكلمنھويىإآؤئء"
)
# Dictionnaire caractere -> index et index -> caractere
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 1-base (0 = blank CTC)
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB)}
IDX_TO_CHAR[0] = ""  # Le blank CTC ne produit aucun caractere

NUM_CLASSES = len(VOCAB) + 1  # +1 pour le blank CTC

# Dimensions de l'image d'entree
IMG_HEIGHT = 32
IMG_WIDTH  = 128


# ------------------------------------------------------------------
# Bloc residuel (inspire de ResNet)
# ------------------------------------------------------------------

def residual_block(x, filters, strides=1, activation="leaky_relu", dropout=0.2):
    """
    Bloc residuel : la sortie = F(x) + x (connexion de saut)
    
    Pourquoi utiliser des blocs residuels ?
      - Evite le probleme de "gradient qui disparait" dans les reseaux profonds
      - Permet d'entrainer des reseaux plus profonds sans degradation
      - Le reseau apprend ce qui CHANGE par rapport a l'entree (le "residu")
    
    Schema :
      x -> [Conv] -> [BN] -> [Act] -> [Conv] -> [BN] -> (+) -> [Act] -> [Dropout]
      x -----------------------------------------[Conv 1x1]--^
                         (connexion de saut, ajustee si necessaire)
    """
    # Chemin principal : 2 convolutions
    out = layers.Conv2D(filters, 3, strides=strides, padding="same",
                        use_bias=False)(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out) if activation == "leaky_relu" else layers.ReLU()(out)
    out = layers.Conv2D(filters, 3, padding="same", use_bias=False)(out)
    out = layers.BatchNormalization()(out)

    # Connexion de saut : ajustement si le nombre de filtres ou le stride change
    skip_conv_needed = (strides != 1 or x.shape[-1] != filters)
    if skip_conv_needed:
        x = layers.Conv2D(filters, 1, strides=strides, use_bias=False)(x)

    # Addition : chemin principal + connexion de saut
    out = layers.Add()([out, x])
    out = layers.LeakyReLU()(out) if activation == "leaky_relu" else layers.ReLU()(out)
    out = layers.Dropout(dropout)(out)
    return out


# ------------------------------------------------------------------
# Modele CRNN complet
# ------------------------------------------------------------------

def build_crnn_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                     num_classes=NUM_CLASSES, dropout=0.2):
    """
    Construit le modele CRNN complet.
    
    Parametres :
      img_height  : hauteur de l'image (32 px)
      img_width   : largeur de l'image (128 px)
      num_classes : taille du vocabulaire + 1 blank CTC
      dropout     : taux de regularisation
    
    Retourne : modele Keras
    
    Dimensions a travers le reseau (batch_size=B) :
      Entree         : (B, 32, 128, 1)   <- image en niveaux de gris
      Apres CNN      : (B,  8,  32, 64)  <- features extraites
      Apres Reshape  : (B, 256,     64)  <- sequence temporelle
      Apres BiLSTM   : (B, 256,    128)  <- contexte bidirectionnel
      Apres Dense    : (B, 256, vocab)   <- probabilites par pas de temps
    """

    # ----- ENTREE -----
    # L'image est en niveaux de gris (1 canal) normalisee entre 0 et 1
    inputs = layers.Input(shape=(img_height, img_width, 1), name="image")

    # Normalisation : pixels 0-255 -> 0-1
    x = layers.Rescaling(scale=1.0 / 255.0)(inputs)

    # ----- CNN : EXTRACTION DES FEATURES VISUELLES -----
    # Le CNN "voit" la plaque et extrait des patterns visuels
    # (courbes des chiffres, forme des lettres arabes, etc.)

    # Bloc 1 : 16 filtres, taille conservee (32x128)
    x = residual_block(x, 16, strides=1, dropout=dropout)

    # Bloc 2 : 16 filtres, resolution reduite (16x64)
    x = residual_block(x, 16, strides=2, dropout=dropout)
    x = residual_block(x, 16, strides=1, dropout=dropout)  # affinage

    # Bloc 3 : 32 filtres, resolution reduite (8x32)
    x = residual_block(x, 32, strides=2, dropout=dropout)
    x = residual_block(x, 32, strides=1, dropout=dropout)  # affinage

    # Bloc 4 : 64 filtres, taille conservee (8x32)
    x = residual_block(x, 64, strides=1, dropout=dropout)
    x = residual_block(x, 64, strides=1, dropout=dropout)  # affinage

    # Forme ici : (B, 8, 32, 64)
    # 8 lignes x 32 colonnes x 64 features

    # ----- TRANSITION CNN -> LSTM -----
    # On "aplatit" les 8 lignes et 64 features en une seule dimension
    # pour obtenir une sequence de 32 pas de temps, chacun avec 8*64=512 features
    # 
    # POURQUOI CE RESHAPE ?
    # Le LSTM attend une sequence 1D de vecteurs.
    # On transforme la carte de features 2D (8x32) en une sequence 1D (32 pas)
    # Chaque pas correspond a une "colonne" de la plaque -> la lecture gauche a droite

    # Permuter les axes : (B, 8, 32, 64) -> (B, 32, 8, 64)
    x = layers.Permute((2, 1, 3))(x)  # echanger hauteur et largeur

    # Aplatir hauteur et features : (B, 32, 8*64) = (B, 32, 512)
    time_steps = x.shape[1]       # 32
    features   = x.shape[2] * x.shape[3]  # 8 * 64 = 512
    x = layers.Reshape((time_steps, features))(x)

    # ----- BiLSTM : COMPREHENSION CONTEXTUELLE -----
    # Le LSTM lit la sequence de gauche a droite ET de droite a gauche
    # pour comprendre le contexte complet avant de predire chaque caractere.
    #
    # Exemple : le "1" dans "90120ا72" est different du "1" seul
    # car le BiLSTM voit ce qui l'entoure dans les deux sens.

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=dropout)
    )(x)
    # Forme : (B, 32, 128)  <- 128 = 64 avant + 64 arriere

    # ----- SORTIE : PROBABILITES PAR PAS DE TEMPS -----
    # Pour chaque pas de temps (colonne de la plaque), on predit
    # la probabilite de chaque caractere du vocabulaire.
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)
    # Forme : (B, 32, num_classes)

    model = keras.Model(inputs=inputs, outputs=output, name="CRNN_Plaque_Marocaine")
    return model


# ------------------------------------------------------------------
# Fonction de perte CTC
# ------------------------------------------------------------------

class CTCLoss(keras.losses.Loss):
    """
    CTC Loss (Connectionist Temporal Classification)
    
    POURQUOI CTC ?
    ==============
    Le probleme : on a une sequence de 32 predictions (une par colonne),
    mais le texte n'a que 7-8 caracteres. Comment les aligner ?
    
    La solution CTC :
      - Introduit un token "blank" (vide) entre les caracteres
      - Permet aux caracteres de s'etendre sur plusieurs colonnes
      - Gere automatiquement l'alignement sans annotation char par char
    
    Exemples d'encodages CTC valides pour "90120" :
      "-9-9-0-1-2-0-" (- = blank)
      "99900011122000"
      "-9-0-1-2-0----"
    Tous ces encodages decodent vers "90120" -> le modele apprend
    a produire n'importe lequel d'entre eux.
    """
    # def __init__(self, name="CTCLoss"):
    #     super().__init__(name=name)
    def __init__(self, name="CTCLoss", **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        batch_size   = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]

        # Longueur des predictions (32 pour toutes les images)
        input_lengths = tf.fill([batch_size], tf.cast(input_length, tf.int32))

        # Longueur reelle de chaque label (sans le padding -1)
        label_lengths = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1
        )

        # Remplacer le padding -1 par 0 (inoffensif pour ctc_loss)
        y_true_masked = tf.where(
            tf.equal(y_true, -1), tf.zeros_like(y_true), y_true
        )

        # tf.nn.ctc_loss attend (batch, time, classes) -> transpose
        logits = tf.math.log(
            tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8
        )
        y_true_masked = tf.cast(y_true_masked, tf.int32)  # <- ajoute cette ligne

        loss = tf.nn.ctc_loss(
            labels=y_true_masked,
            logits=logits,
            label_length=label_lengths,
            logit_length=input_lengths,
            blank_index=0,        # <- blank explicitement a l'index 0
            logits_time_major=True,
        )
        return tf.reduce_mean(loss)

# ------------------------------------------------------------------
# Decodage CTC : probabilites -> texte
# ------------------------------------------------------------------

def decode_predictions(y_pred, idx_to_char=IDX_TO_CHAR, blank_idx=0):
    """
    Decodage CTC greedy COHERENT avec blank_index=0.

    Pourquoi ne pas utiliser ctc_decode de Keras ?
    -> ctc_decode suppose toujours blank = DERNIER index.
    -> On entraine avec blank = index 0 (tf.nn.ctc_loss blank_index=0).
    -> Il faut donc decoder manuellement pour rester coherent.

    Algorithme :
      1. argmax a chaque pas de temps -> sequence d'indices
      2. Supprimer les repetitions consecutives
      3. Supprimer les blanks (index 0)
    """
    # y_pred : (batch, time_steps, num_classes)
    best_path = np.argmax(y_pred, axis=-1)  # (batch, time_steps)

    results = []
    for path in best_path:
        # Etape 2 : supprimer les repetitions consecutives
        collapsed = [path[0]]
        for idx in path[1:]:
            if idx != collapsed[-1]:
                collapsed.append(idx)

        # Etape 3 : supprimer les blanks et convertir en texte
        text = "".join(
            idx_to_char.get(int(idx), "")
            for idx in collapsed
            if int(idx) != blank_idx
        )
        results.append(text)

    return results


# ------------------------------------------------------------------
# Test rapide du modele
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[INFO] Vocabulaire ({len(VOCAB)} classes + 1 blank CTC = {NUM_CLASSES}) :")
    print(f"       {VOCAB}")
    print()

    model = build_crnn_model()
    model.summary()

    print(f"\n[INFO] Parametres totaux : {model.count_params():,}")
    print(f"[INFO] Forme entree      : {model.input_shape}")
    print(f"[INFO] Forme sortie      : {model.output_shape}")
    print(f"       -> {model.output_shape[1]} pas de temps")
    print(f"       -> {model.output_shape[2]} classes par pas de temps")
