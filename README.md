# Quasar Framework

**Quasar Framework** est un framework simple de réseau de neurones, écrit en Python, qui permet de construire, d'entraîner et d'évaluer des modèles de réseaux de neurones. Il utilise des concepts fondamentaux tels que la propagation avant, la rétropropagation et le calcul des erreurs.

---

## 🚀 Fonctionnalités

- ✅ **Ajout de couches personnalisées** au réseau.
- ✅ Support des **fonctions de perte** :
  - Erreur Absolue Moyenne (**MAE**)
  - Erreur Quadratique Moyenne (**MSE**)
- ✅ **Propagation avant** et **rétropropagation** pour l'entraînement des modèles.
- ✅ Mise à jour des poids et des biais à l'aide de la **descente de gradient**.

---

## 🛠️ Installation

Pour utiliser ce framework, assurez-vous d'avoir **Python 3.x** installé sur votre machine. Ensuite, clonez le dépôt et installez les dépendances nécessaires :

```bash
git clone <lien_du_depot>
cd Quasar Framework
pip install numpy
```

---

## 📘 Utilisation

Voici un exemple de base pour créer et entraîner un réseau de neurones :

```python
import numpy as np
from network import Network

# Exemple de données
input_dim = 2
output_dim = 1
train_data = np.array([[1, 1]])  # Données d'entrée
Y_true = np.array([[0.5]])       # Données de sortie réelles

# Initialiser le modèle et ajouter des couches
my_model = Network()
my_model.add(64)             # Première couche cachée avec 64 neurones
my_model.add(output_dim)     # Couche de sortie avec 1 neurone

# Ajuster le modèle avec la dimension d'entrée spécifiée
my_model.fit(input_dim)

# Entraîner le modèle
my_model.train(train_data, Y_true, error='mse', epochs=100, lr=0.001)

# Afficher la sortie après l'entraînement
print(my_model.forward(train_data))
```

---

## 📂 Structure du code

Le framework est organisé en plusieurs fichiers :

- **`network.py`** : Contient la classe `Network`, qui gère l'architecture et l'entraînement du réseau.
- **`layer.py`** : Contient la classe `Layer`, qui représente une couche individuelle du réseau.
- **`error.py`** : Contient la classe `Error`, qui calcule les erreurs et les gradients.

---

## 🤝 Contributions

Les contributions sont les bienvenues ! 🎉  
Si vous souhaitez contribuer à ce projet, veuillez créer une **issue** ou soumettre une **pull request**.

---
