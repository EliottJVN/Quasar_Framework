# Quasar Framework

**Quasar Framework** est un framework de réseau de neurones, écrit en Python, qui permet de construire, d'entraîner des modèles de réseaux de neurones. Il utilise des concepts fondamentaux tels que la propagation avant, la rétropropagation et le calcul des erreurs.

---

## 🚀 Fonctionnalités

- ✅ **Ajout de couches personnalisées** au réseau.
- ✅ Support des **couches d'activation** :
  - Fonction d'activation `relu`
  - Fonction d'activation `sigmoid`
  - Fonction d'activation `tanh`
  - Fonction d'activation `softmax`
- ✅ Support des **fonctions de perte** :
  - Erreur Absolue Moyenne `mae`
  - Erreur Quadratique Moyenne `mse`
- ✅ **Entraînement** des réseaux de neurones:
  - **Propagation avant** et **rétropropagation** pour l'entraînement des modèles.
  - Mise à jour des poids et des biais à l'aide de la **descente de gradient**.
- ✅ **Sauvegarder & Charger** des modèles déjà entrainés. Le format de sauvegarde des modèles est `json`.

---

## 🛠️ Installation

Pour utiliser ce framework, assurez-vous d'avoir **Python 3.x** installé sur votre machine. Ensuite, clonez le dépôt et installez les dépendances nécessaires :

```bash
git clone git@github.com:EliottJVN/Quasar_Framework.git
cd Quasar_Framework
pip install numpy json
```

---

## 📘 Utilisation

Voici un exemple de base pour créer et entraîner un réseau de neurones :

```python
import numpy as np
from network import Network

# Example on AND gate
input_dim = 2
output_dim = 1

# AND Gate.
train_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_true = np.array([[0], [0], [0], [1]])

# Initialize the model and add layers
my_model = Network()
my_model.add(4)   # First hidden layer with 4 neurons
my_model.add('relu')
my_model.add(1)   # Output layer with 1 neuron
my_model.add('sigmoid') 

# Fit the model with the specified input dimension
my_model.fit(input_dim)

# Train the model
my_model.train(train_data, Y_true, error='mse', epochs=5000, lr=0.01)

print(my_model.forward(train_data).round())
```

---

## 📂 Structure du code

Le framework est organisé en plusieurs fichiers :

- **`network.py`** : Contient la classe `Network`, qui gère l'architecture et l'entraînement du réseau.
- **`layer.py`** : Contient la classe `Layer`, qui représente une couche individuelle du réseau.
- **`activationlayer.py`** : Contient la classe `ActivationLayer`, qui représente une couche d'activation du réseau & contient les fonctions d'activation utilisées dans le réseau
- **`error.py`** : Contient la classe `Error`, qui calcule les erreurs et les gradients.

---

## 🤝 Contributions

Les contributions sont les bienvenues ! 🎉  
Si vous souhaitez contribuer à ce projet, veuillez créer une **issue** ou soumettre une **pull request**.

---
