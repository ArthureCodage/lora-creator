<div align="center">

# ◆ LoRA Creator

**Une interface minimaliste pour entraîner des LoRA avec [kohya_ss](https://github.com/bmaltais/kohya_ss), sans régler 25 paramètres à chaque fois.**

Pensée pour les petites cartes graphiques (testée sur une RTX 2060 6 Go) · SD 1.5 et SDXL / Pony · Windows

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![kohya_ss](https://img.shields.io/badge/moteur-kohya__ss%20sd--scripts-8b7cf6)
![Windows](https://img.shields.io/badge/Windows-10%20%2F%2011-0078D6?logo=windows&logoColor=white)
![License](https://img.shields.io/badge/licence-MIT-green)

<img src="docs/images-legendes.jpg" alt="Écran principal de LoRA Creator" width="800">

</div>

---

## Pourquoi ?

La GUI de kohya_ss est puissante, mais elle expose des centaines de réglages. Pour entraîner un LoRA « normal »,
on finit toujours par recharger un preset, vérifier les chemins, recalculer les répétitions…

**LoRA Creator** garde kohya comme moteur, mais fixe une fois pour toutes les réglages techniques adaptés
à ta machine. À l'écran, il ne reste que ce qui compte :

> **le type de LoRA · un nom · des images · un modèle · une qualité → Lancer.**

## Fonctionnalités

| | |
|---|---|
| 🎯 **Type demandé en popup** | Personnage, style ou objet/concept : chaque type a ses presets (dimension, nombre de pas, prompts de test). |
| 🖼️ **Préparation des images** | Glisser-déposer de fichiers ou d'un dossier entier, import par chemin. Images redressées (EXIF), converties, réduites, doublons ignorés. Pas de recadrage : le *bucketing* de kohya s'en charge. |
| 🏷️ **Légendes automatiques** | Tags WD14 (swinv2 v3, sur GPU via ONNX) ou phrases BLIP. Éditeur de légendes en grille, suppression d'un tag partout en un clic. Le mot-clé est ajouté automatiquement (`caption_prefix`), les fichiers ne sont pas modifiés. |
| ⏱️ **Estimation du temps** | Affichée pour SD 1.5 **et** SDXL avant de lancer, puis recalibrée automatiquement à partir de tes vrais entraînements. |
| 🎚️ **3 niveaux de qualité** | Rapide / Équilibré / Qualité. Les répétitions et epochs sont calculées selon la taille du dataset. Un volet « Avancé » replié permet de tout ajuster. |
| 📈 **Suivi en direct** | Progression, temps restant, loss, images de test à chaque sauvegarde, journal kohya, bouton d'arrêt. |
| 📦 **Copie vers Forge / A1111** | Le LoRA fini est copié dans `models/Lora` avec son aperçu et un `.json` (mot-clé, poids conseillé, version SD). Aucun LoRA existant n'est jamais écrasé. |
| 🗂️ **Historique** | Chaque entraînement est gardé (réglages, durée, aperçus, fichiers par epoch) avec un bouton **Relancer avec les mêmes réglages**. |

<table>
<tr>
<td><img src="docs/popup-type.jpg" alt="Choix du type de LoRA"></td>
<td><img src="docs/modele-estimation.jpg" alt="Choix du modèle avec estimation du temps"></td>
</tr>
<tr>
<td align="center"><sub>Le type est demandé à chaque nouveau LoRA</sub></td>
<td align="center"><sub>Temps estimé pour SD 1.5 et SDXL / Pony</sub></td>
</tr>
<tr>
<td colspan="2"><img src="docs/entrainement.jpg" alt="Suivi de l'entraînement"></td>
</tr>
<tr>
<td colspan="2" align="center"><sub>Suivi de l'entraînement, images de test et fichiers par epoch</sub></td>
</tr>
</table>

## Installation

**Prérequis :** Windows, une carte NVIDIA et [kohya_ss](https://github.com/bmaltais/kohya_ss) déjà installé (testé avec la v26).
LoRA Creator n'installe rien : il utilise le Python et les paquets de kohya (FastAPI, Pillow, onnxruntime…).

```bash
git clone https://github.com/ArthureCodage/lora-creator.git
```

1. Double-clic sur **`Lancer LoRA Creator.bat`**.
   Il cherche kohya dans `%USERPROFILE%\kohya_ss`. Sinon, définis la variable d'environnement `KOHYA_DIR`.
2. Au premier lancement, `config.json` est créé depuis [`config.example.json`](config.example.json).
   Adapte les chemins, puis relance :

| Clé | Rôle |
|---|---|
| `kohya_dir` | Dossier d'installation de kohya_ss |
| `data_dir` | Où ranger datasets, entraînements et LoRA produits |
| `forge_lora_dir` | Dossier `models/Lora` de Forge / A1111 (copie automatique) |
| `model_dirs` | Dossiers où chercher les checkpoints `.safetensors` (SD 1.5 et SDXL détectés automatiquement) |
| `port` | Port de l'interface (7870 par défaut) |
| `tagger_repo` | Modèle WD14 utilisé pour les tags |

L'interface s'ouvre sur <http://127.0.0.1:7870>. Pour arrêter l'appli, ferme la fenêtre noire.

## Utilisation

1. **Nouveau LoRA** : choisis le type, donne un nom. Le mot-clé déclencheur est proposé automatiquement.
2. **Images** : glisse tes images, clique sur **Légendes auto**, corrige si besoin.
3. **Modèle** : choisis le checkpoint. Le temps estimé est affiché pour chaque famille.
4. **Qualité** : Rapide pour tester, Équilibré par défaut, Qualité pour le rendu final.
5. **Lancer** : suis la progression dans l'onglet *Entraînement*. Le LoRA arrive tout seul dans Forge.

## Réglages fixés

Choisis pour tenir dans **6 Go de VRAM** :

- **Communs** : AdamW8bit, `cosine_with_restarts`, fp16, xformers, gradient checkpointing, cache des latents sur disque,
  `min_snr_gamma = 5`, bucketing, environ 5 sauvegardes par entraînement pour pouvoir choisir le meilleur epoch.
- **SD 1.5** : 512 px, clip skip 2, LR U-Net 1e-4 / text encoder 5e-5, shuffle caption, dim 32 / alpha 16 (style : 64/32).
- **SDXL / Pony** : 1024 px, U-Net seul, `fp8_base`, cache des sorties du text encoder, `no_half_vae`, dim 32 / alpha 16.
  Pour les modèles Pony, des tags `score_*` sont ajoutés aux prompts de test.

Nombre de pas visés :

| | Rapide | Équilibré | Qualité |
|---|---:|---:|---:|
| SD 1.5 · personnage | 700 | 1 400 | 2 400 |
| SD 1.5 · style | 1 000 | 1 800 | 3 000 |
| SD 1.5 · objet | 600 | 1 200 | 2 000 |
| SDXL · personnage | 500 | 1 000 | 1 600 |
| SDXL · style | 700 | 1 300 | 2 000 |
| SDXL · objet | 450 | 900 | 1 400 |

> **SDXL sur 6 Go** : l'entraînement fonctionne (≈ 4,5 s/pas à 1024 px sur RTX 2060), mais une image de test prend
> environ 9 minutes, car la VRAM déborde dans la RAM. Les images de test sont donc désactivées par défaut en SDXL.

Tous ces réglages sont dans [`app/engine.py`](app/engine.py), en haut du fichier.

## Organisation des données

```
data_dir/
├── datasets/<nom>/        images préparées + légendes .txt
├── runs/<date-nom>/       un dossier par entraînement
│   ├── run.json           historique (réglages, durée, mesures)
│   ├── config.toml        arguments passés à sd-scripts
│   ├── dataset.toml       config du dataset (répétitions, mot-clé)
│   ├── train.log          journal complet
│   └── output/            LoRA par epoch + sample/
├── LoRA/                  copie de chaque LoRA final
└── models/wd14/           modèle de tagging (téléchargé au 1er usage)
```

## Structure du code

```
app/
├── engine.py      presets, estimation, datasets, lancement de sd-scripts, historique, copie Forge
├── server.py      API FastAPI + fichiers statiques
└── static/        interface (HTML / CSS / JS vanilla, aucune étape de build)
```

## Licence

[MIT](LICENSE). LoRA Creator n'est pas affilié à kohya_ss. Il se contente de piloter ses scripts.
