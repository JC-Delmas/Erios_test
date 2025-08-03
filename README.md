# Test technique pour le projet ERIOS avec RAG et LLM (Google Colab)

Ce notebook Google Colab propose la mise en place d’un assistant médical local, capable de répondre à des questions en français à partir d’un document médical importé (.docx), en utilisant un grand modèle de langage (LLM) open source, et une approche RAG (Retrieval Augmented Generation). L’intégralité du traitement s’effectue sur CPU, sans recours au cloud externe.

Pour accéder au Google Colab, suivez ce lien : https://colab.research.google.com/drive/106JRPfHVvf6qtBRSpe8oFWPV3lwCISLb?usp=sharing

---

## Fonctionnalités principales

- **Import d’un document médical (.docx)** qui sert de base de connaissances.
- **Extraction et indexation du contenu** pour permettre une recherche intelligente.
- **Recherche sémantique** (vectorielle) des passages les plus pertinents pour chaque question posée.
- **Génération de réponses** en français à l’aide du modèle LLM Mistral 7B (quantifié pour le CPU).
- **Citations automatiques** des sources utilisées dans chaque réponse.
- **Boucle interactive** pour poser des questions médicales en continu.

---

## Étapes du Notebook

### 1. Installation des dépendances

Les bibliothèques nécessaires sont installées automatiquement :
- `llama-cpp-python` (pour exécuter le LLM localement)
- `sentence-transformers` (pour les embeddings)
- `faiss-cpu` (pour l’indexation vectorielle)
- `python-docx` (pour lire les fichiers .docx)
- `tqdm` (progression)

### 2. Téléchargement du modèle et import du document

- Téléchargement d’un modèle LLM quantifié (Mistral 7B Instruct, 3,5 Go) pré-entraîné.
- Import du document médical via l’interface Colab.

### 3. Préparation et indexation des données

- Extraction des paragraphes du document.
- Découpage en « passages » pour optimiser la pertinence des recherches.
- Génération d’« embeddings » pour chaque passage via SentenceTransformers.
- Construction d’un index vectoriel (FAISS) pour une recherche rapide des passages similaires à la question posée.

### 4. Chargement du modèle LLM

- Chargement du modèle Mistral 7B quantifié sur le CPU.
- Configuration adaptée à la machine Colab (nombre de threads, taille de contexte).

### 5. Fonction de réponse avec citations

- Recherche vectorielle des passages les plus pertinents par rapport à la question.
- Filtrage par mots-clés et nombres (utile pour les médicaments ou dosages).
- Construction du prompt pour le LLM incluant le contexte extrait.
- Génération de la réponse en streaming, avec citations précises des sources utilisées.

### 6. Boucle interactive

- Permet de poser des questions successives à l’assistant, qui répond de façon concise, cite les sources, et termine toujours par le symbole `<FIN>`.

---

## Prérequis

- **Google Colab** (exécution sur CPU)
- **Fichier .docx** à importer (contenant des données médicales ou scientifiques)

---

## Utilisation

1. Exécutez les cellules dans l’ordre.
2. Lors de l’exécution de la cellule d’import, chargez votre fichier `.docx`.
3. Lorsque la boucle interactive démarre, posez vos questions médicales directement dans l’interface.
4. Pour quitter, appuyez simplement sur Entrée sans saisir de texte.

---

## Utilisation

```
❓ Posez votre question médicale (Entrée pour quitter) : Quels sont les effets secondaires du médicament X ?
▶️ génération : ... (réponse générée avec citations des sources du document)
---
Sources :
[source 3] « Extrait du passage concerné du document… »
```

## Exemples de prompt
```
❓ Posez votre question médicale (Entrée pour quitter) : Donnez trois exemples de comorbidités ou facteurs de risque qui font choisir une antibiothérapie probabiliste élargie (tableau 2).

▶️ génération : 1. Immunodépression [source 5]
2. Néoplasie active [source 5]
3. BPCO sévère ou insuffisance respiratoire chronique [source 13] 
⏱ 119.3 s

---
Sources :
[source 5] « • Néoplasie active • Immunodépression • BPCO sévère ou insuffisance respiratoire chronique 6. Antibiothérapie probabiliste (adulte hospitalisé) 🔹 PAC non grave, sans comorbidité :… »
[source 13] « - vie en institution. Décision d’orientation : - Âge < 65 ans sans ou avec un seul facteur de risque → prise en charge ambulatoire - Âge < 65 ans et ≥ 2 facteurs de risque →… »
```

```
❓ Posez votre question médicale (Entrée pour quitter) : Quelles sont les trois indications d’un scanner thoracique low-dose selon la synthèse ?

▶️ génération : 1. Doute diagnostique, 2. Pas d'amélioration après 72 heures, 3. Facteur de risque de cancer (> 50 ans et > 20 PA) après 2 mois. [source 9, 0] 
⏱ 139.9 s

---
Sources :
[source 9] « • Réévaluation à J4 ; décroissance progressive • Durée totale 8–14 jours 9. Examens complémentaires • 1ère intention : Radiographie thoracique ou échographie pulmonaire • TDM… »

```

```
❓Posez votre question médicale (Entrée pour quitter) : Quelle est la posologie d’amoxicilline–acide clavulanique recommandée en traitement probabiliste d’une PAC grave chez l’adulte hospitalisé, et quelle est la durée maximale autorisée ?

▶️ génération : 2 g IV ou PO toutes 8 h [source 5]. La durée maximale autorisée n'est pas spécifiée dans la source [source 5]. Cependant, la durée de traitement standard pour une PAC grave est de 7 à 10 jours [source 1]. 
⏱ 219.3 s

---
Sources :
[source 1] « 2. Points clés • Critères de stabilité : apyrexie, stabilité hémodynamique, pouls ≤ 100/min, FR ≤ 24/min, SpO₂ ≥ 90 % (AA). • Si critères atteints à J3 → 3 jours d'antibiotiques ;… »
[source 5] « • Néoplasie active • Immunodépression • BPCO sévère ou insuffisance respiratoire chronique 6. Antibiothérapie probabiliste (adulte hospitalisé) 🔹 PAC non grave, sans comorbidité :… »

```

## Remarques

- **Aucune donnée n’est transmise à des serveurs externes** : tout est exécuté localement sur la machine Colab.
- L’assistant répond toujours en français, de manière concise, sans jamais poser de nouvelle question.

---

## Limites

- Dépend de la qualité et de l’exhaustivité du document importé.
- La vitesse de génération peut varier selon la taille du modèle et la puissance CPU de la machine Colab (compter environ 200 secondes par réponse avec ce prototype).
- La restitution des sources peuvent manquer ou être incomplètes selon le contexte, bien que le modèle s'appuie sur le document pour répondre.
- Pas de réelle API disponible pour l'instant.
- Non destiné à un usage médical réel sans validation humaine.

---

## Auteur

Jean-Charles DELMAS
jean-charles.delmas@chu-montpellier.fr
Bioinformaticien - UMR1183 - INSERM/CHU de Montpellier
Plateforme LGMMR pour Laboratoire de Génétique Moléculaire de Maladies Rares, Secteur des Maladies Rares et Auto‑Inflammatoires.

Ce projet a été réalisé avec l'assistance d'une IA (certaines parties de code, commentaires et linting).
