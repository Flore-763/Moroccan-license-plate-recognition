import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_report.csv")

classes = [
    '0','1','2','3','4',
    '5','6','7','8','9',
    'أ','ب','د','و','ھ'
]

y_true = []
y_pred = []

for gt, pred in zip(df["true"], df["pred"]):

    gt = str(gt)
    pred = str(pred)

    L = min(len(gt), len(pred))

    for i in range(L):
        y_true.append(gt[i])
        y_pred.append(pred[i])

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=classes
)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Prédiction")
plt.ylabel("vraie immatriculation")
plt.title("Matrice de confusion des 15 caractères")

plt.show()

accuracy = np.mean(np.array(y_true) == np.array(y_pred))  # accuracy

print("Accuracy: ",accuracy)

correct = (  #rappel
    df["true"] == df["pred"]
).mean()

print("Rappel: ",correct)

print(  #Fscore
    classification_report(
        y_true,
        y_pred,
        labels=classes
    )
)

