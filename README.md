# Projets Deep Learning — Template

Installez l’environnement (via `pip` ou `conda`) et utilisez les scripts dans `src/`.
**Les consignes détaillées sont sur le site du cours.**

## Installation rapide
```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# conda (option)
# conda create -n dl-projets python=3.10 -y
# conda activate dl-projets
# pip install -r requirements.txt
````



# Activer l'environnement d'exécution

mamba activate csc8607

# Allocation des ressources

salloc --time=01:00:00 --gres=gpu:1 --cpus-per-task=2 --mem=8G

srun --pty bash

## TensorBoard pour visualiser les courbes

### Côté serveur

tensorboard --logdir=runs --port=6006 --bind_all

### En local

ssh -L 6006:arcadia-slurm-node-2:6006 tsp-client

* Modifiez `configs/config.yaml` selon votre projet.
* Implémentez les fonctions dans `src/`.
* Utilisez le `Makefile` pour lancer les étapes (ou exécutez les modules Python directement).

> Reportez-vous au site du cours pour les règles, livrables et barème.
