# Traitement automatique des candidatures sur parcoursup

Ce projet vise dans un premier temps à automatiser l'étude des lettres de motivation des candidatures universitaires en utilisant un réseau de neurones. 

Nous utilisons le modèle Flaubert pour la classification de texte afin de prédire les scores des lettres, qui sont ensuite utilisés pour le classement.

### Prérequis

- Python 3.7 ou supérieur
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- Flaubert

### Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/sohaibafifi/parcoursup-ia.git
```

### Installez les paquets requis :
```
pip install -r requirements.txt
```

### Utilisation

Placez vos données brutes dans le répertoire data/raw/. (un fichier excel avec deux colonnes, motivations et scores)
Mettez à jour le script src/main.py pour charger et prétraiter vos données.
Exécutez le script principal pour entraîner et évaluer le modèle :
```
python src/main.py
```
