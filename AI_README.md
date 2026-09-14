# Lalachante — Prototype V4 Content-Based + IA contextuelle

Cette V4 se pose **par-dessus ton projet actuel**. Elle ne remplace pas tes fichiers MusicBrainz/ListenBrainz existants et ne nécessite pas de refaire le matching.

## Objectif

Corriger les limites d'un moteur basé uniquement sur :

- `danceability`
- `energy`
- `valence`
- `tempo`
- `acousticness`
- `instrumentalness`

en ajoutant un contexte appris par un modèle d'embeddings multilingue :

- artiste ;
- genre ;
- pays/zone MusicBrainz ;
- année de sortie.

La popularité brute n'entre **jamais** directement dans le score. Lorsque ListenBrainz est disponible, elle est transformée en **percentile à l'intérieur du pays de l'artiste**.

Ainsi, un marché très grand ne reçoit pas automatiquement un avantage sur un marché plus petit.

---

# 1. Fichiers à copier

Copie les fichiers de ce ZIP dans ton projet existant en respectant les dossiers :

```text
lalachante_recommender_complete/
│
├── app_ai.py
├── requirements_ai.txt
│
├── src/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── embedding_store.py
│   │   └── build_context_embeddings.py
│   │
│   ├── recommender/
│   │   └── engine_ai.py
│   │
│   └── evaluation/
│       └── evaluate_ai.py
│
└── scripts/
    └── run_ai_prototype.ps1
```

Ton ancien `app.py` et ton ancien `engine.py` restent intacts.

---

# 2. Prérequis données

Avant de construire les embeddings, ton fichier :

```text
data/processed/recommendation_dataset.csv
```

doit idéalement contenir :

```text
track_id
track_name
artists
track_genre

danceability
energy
valence
tempo
acousticness
instrumentalness

recording_mbid
primary_artist_mbid
artist_country
artist_area_name
release_year

lb_recording_listener_count
lb_artist_listener_count
```

Toutes les colonnes MusicBrainz/ListenBrainz sont facultatives : le moteur redistribue automatiquement le poids lorsqu'une donnée manque.

Mais plus `artist_country` est renseigné, mieux le système peut corriger les incohérences de genre/pays.

---

# 3. Installer l'IA

Dans ton environnement `.venv` :

```powershell
python -m pip install -r requirements_ai.txt
```

Le modèle choisi est :

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Il est multilingue et fonctionne sur CPU. Il sera téléchargé une seule fois lors de la première construction des embeddings.

Aucune API payante n'est utilisée.

---

# 4. Construire les embeddings

```powershell
python -m src.ai.build_context_embeddings
```

Le script transforme chaque morceau en texte contextuel, par exemple :

```text
Track: Super Shy.
Artist: NewJeans.
Genre: k-pop.
Artist country: South Korea.
Artist area: Seoul.
Release year: 2023.
```

Puis le modèle produit un vecteur sémantique.

Les fichiers générés sont :

```text
data/processed/ai/
├── context_embeddings.npy
└── context_embeddings_meta.json
```

Les embeddings sont normalisés et sauvegardés en `float32`.

Le programme enregistre aussi une empreinte des `track_id`. Si tu reconstruis plus tard `recommendation_dataset.csv`, l'application détectera que les embeddings ne correspondent plus au dataset et te demandera simplement de les reconstruire.

---

# 5. Lancer le prototype IA

```powershell
streamlit run app_ai.py
```

---

# 6. Comment fonctionne le score

Cette version n'utilise pas une formule rigide du type :

```text
60 % audio
15 % genre
15 % pays
...
```

Elle crée des **poids adaptatifs** en fonction de ce que le profil permet réellement d'inférer.

Exemple :

```text
Utilisateur aime :
- NewJeans → KR
- BLACKPINK → KR
- LE SSERAFIM → KR

Couverture pays = 100 %
Concentration pays = 100 %

=> le signal pays devient important.
```

Mais :

```text
Utilisateur aime :
- artiste KR
- artiste US
- artiste FR
- artiste BR

=> profil géographique dispersé
=> le poids pays diminue automatiquement.
```

Les signaux utilisés sont :

```text
audio_similarity
ai_context_similarity
genre_affinity
country_affinity
era_affinity
local_popularity
novelty
```

Le moteur normalise les poids disponibles pour chaque candidat. Une chanson sans année ou sans pays n'est donc pas automatiquement pénalisée à zéro.

---

# 7. Popularité sans biais de taille de marché

Mauvaise approche :

```text
nombre brut d'auditeurs IN
vs
nombre brut d'auditeurs KR
```

Approche V4 :

```text
percentile de popularité à l'intérieur de IN
vs
percentile de popularité à l'intérieur de KR
```

Exemple :

```text
Artiste indien : top 8 % de son pays  → 0.92
Artiste coréen : top 6 % de son pays → 0.94
```

La population brute du marché ne donne plus directement un avantage.

---

# 8. Pourquoi un artiste indien étiqueté `k-pop` descend

Le dataset Spotify peut contenir une taxonomie imparfaite.

Un candidat peut avoir :

```text
genre = k-pop
audio similarity = 0.98
country = IN
```

alors que l'utilisateur aime surtout des artistes :

```text
KR + KR + KR
```

Avec la V4 :

```text
Audio                         élevé
Genre                         élevé
Contexte IA                   plus faible
Affinité pays                 faible
Popularité brute              ignorée
Popularité locale             comparable entre marchés
```

Le morceau peut toujours être recommandé s'il est vraiment pertinent, mais il n'est plus favorisé uniquement parce qu'il porte l'étiquette `k-pop` ou possède un volume d'écoute brut élevé.

Il n'existe aucune règle :

```python
if country == "IN":
    score -= ...
```

Le système fonctionne de la même manière pour tous les pays.

---

# 9. Onboarding amélioré

Même avant que l'utilisateur ait choisi des chansons, `representative_tracks()` analyse la distribution géographique réelle du genre.

Si un genre possède un pays dominant très net dans les métadonnées MusicBrainz, les 10 morceaux proposés pour définir le profil sont prioritairement choisis parmi ce contexte dominant.

Cette règle est **apprise depuis les données du catalogue**, pas codée sous forme :

```python
"k-pop" -> "KR"
```

---

# 10. Evaluation

Après la construction des embeddings :

```powershell
python -m src.evaluation.evaluate_ai
```

Le script compare les trois modes :

```text
proche
equilibre
decouverte
```

et mesure notamment :

- similarité audio ;
- similarité contextuelle IA ;
- diversité artistes ;
- diversité genres ;
- cohérence pays lorsqu'un pays de référence existe.

Il s'agit d'une évaluation de diagnostic en absence de vraies interactions utilisateur. Lorsque Lalachante disposera de clics, likes, skips et écoutes, la prochaine étape pourra être un Learning-to-Rank supervisé.

---

# Pipeline final

```text
Spotify
   ↓
features audio
   │
   ├─────────────────────────────┐
   │                             │
   ↓                             ↓
cosine audio               MusicBrainz
                           genre/pays/année
                                 │
                                 ↓
                       Sentence Transformer
                                 │
                                 ↓
                         embeddings contexte
   │                             │
   └──────────────┬──────────────┘
                  ↓
         profil utilisateur
                  │
      ┌───────────┼──────────────┐
      ↓           ↓              ↓
 genre         pays        popularité locale
      │           │              │
      └───────────┴───────┬──────┘
                          ↓
                  scoring adaptatif
                          ↓
                    re-ranking MMR
                          ↓
                       TOP-N
```
