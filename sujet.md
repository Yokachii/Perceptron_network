# TP — Implémentation d'un perceptron en C

**Niveau** : L2
**Durée estimée** : plusieurs semaines de travail (en autonomie, avec séances d'encadrement)
**Langage** : C (norme C99)

## 1. Objectifs

À l'issue de ce TP, vous serez capable de :
- expliquer le fonctionnement d'un neurone formel (perceptron) à sortie sigmoïde ;
- implémenter l'algorithme d'apprentissage par descente de gradient sur un seul neurone ;
- manipuler tableaux, pointeurs, nombres eprésentés en virgule flottante, fonctions et lecture de fichiers en C ;
- évaluer un classifieur binaire au-delà de la simple accuracy, à l'aide d'une matrice de
  confusion (précision, rappel, F1-score), et choisir la bonne métrique selon le coût des
  erreurs dans un cas d'usage donné ;
- comprendre pourquoi l'échelle des données d'entrée influence la stabilité de l'apprentissage,
  et savoir normaliser un jeu de données pour s'en affranchir.

## 2. Mise en situation : le secret du vieux sage

> **Avertissement** : ce scénario est entièrement fictif et les données sont synthétiques
> (générées artificiellement pour ce TP). Ce n'est en aucun cas une méthode réelle
> d'identification des champignons comestibles. **Ne vous fiez jamais à une heuristique simple
> pour juger si un champignon réel est comestible** : l'intoxication par champignon peut être
> mortelle, et seule l'expertise d'un mycologue ou d'un pharmacien formé peut trancher. Ce
> scénario sert uniquement d'habillage pédagogique à un problème de classification binaire.

Le vieux sage du village est mort en emportant avec lui un secret précieux pour toute la
communauté du peuple Mushroom : il était le seul à savoir reconnaître, d'un coup d'œil, si un
champignon était comestible ou non. Il est mort avant d'avoir pu transmettre son savoir.

Le peuple Mushroom dépend des champignons comme unique source de nourriture. Une erreur de
classification n'est donc jamais anodine :
- déclarer comestible un champignon qui ne l'est pas peu provoquer une intoxication, potentiellement mortelle ;
- déclarer non comestible un champignon qui l'est, alors un champignon sain est jeté : une perte de
  nourriture, mais sans danger pour personne.

Heureusement, il reste deux paniers de champignons déjà triés par le sage avant sa mort : un
panier de champignons **comestibles**, un panier de champignons **non comestibles**. C'est le
fichier `data/champignons.csv` : 40 champignons déjà étiquetés (`label = 0` → comestible,
`label = 1` → non comestible), sur lesquels vous allez entraîner votre perceptron pour qu'il
apprenne, à votre tour, à faire ce que le sage faisait.

D'après les carnets du sage (retrouvés après sa mort, mais hélas incomplets - on n'a que les
mesures et la façon dont il les convertit en nombre, pas la règle de décision), on sait que le sage observait toujours les quatre mêmes critères sur un champignon, chacun mesuré sur une échelle de 0 à 255 :

| Critère | Colonne | Description |
|---|---|---|
| Taille du chapeau | x1 | diamètre du chapeau |
| Hauteur du pied | x2 | hauteur du pied |
| Intensité de la couleur | x3 | indice colorimétrique du chapeau (plus la valeur est haute, plus la couleur est vive) |
| Taille du pied | x4 | diamètre du chapeau | 

C'est à vous de retrouver cette règle en entraînant un perceptron sur les données déjà triées.

### Votre mission

1. Entraînez votre perceptron sur une partie du panier trié (`dataset_split`, ex: 80% train).
2. Les critères du sage vont jusqu'à 255 : normalisez vos données (`dataset_normalize`,
   étape 7) avant l'apprentissage, pour pouvoir garder un `learning_rate` raisonnable et stable.
3. Validez ses performances sur les champignons mis de côté (les 20% de test) : calculez la
   matrice de confusion, l'accuracy, la précision, le rappel et le F1-score (étape 8).

## 3. Contexte théorique : le perceptron

Un perceptron à sortie sigmoïde calcule, pour une entrée `x = (x1, ..., xn)` :

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
y = sigmoid(z) = 1 / (1 + exp(-z))
```

`y` est un réel entre 0 et 1, interprété comme une probabilité d'appartenance à la classe 1.
Pour classer un exemple, on applique un seuil : classe 1 si `y >= 0.5`, classe 0 sinon.

### Apprentissage

On entraîne le perceptron en lui présentant des exemples `(x, label)` avec `label` égal à 0 ou 1.
À chaque exemple, on calcule l'erreur entre la sortie prédite et le label, puis on ajuste les
poids et le biais dans la direction qui réduit cette erreur (descente de gradient).

Avec une fonction de coût de type cross-entropy, on peut montrer que la dérivée par rapport aux
poids se simplifie en `(label - y)`, ce qui donne la règle de mise à jour suivante, pour un
taux d'apprentissage `lr` :

==> Demonstration à faire pour la prochaine fois: utiliserz les dérivés partielles 

```
pour chaque poids i :  w_i <- w_i + lr * (label - y) * x_i
biais :                b   <- b   + lr * (label - y)
```

Une **époque** correspond à un passage complet sur tous les exemples du dataset.

### Pourquoi normaliser les entrées ?

Regardez à nouveau la règle de mise à jour : le pas appliqué à `w_i` à chaque exemple est
`lr * erreur * x_i`. Sa taille dépend du produit de `lr` **et** de `x_i`, pas de `lr` seul.

Sur `et.csv`/`ou.csv`, `x_i ∈ {0,1}` : avec `lr = 0.1`, le pas maximal par exemple est de l'ordre
de `0.1`. Mais sur un dataset où `x_i` va jusqu'à 255 (comme `champignons.csv`), garder
`lr = 0.1` donnerait un pas jusqu'à `0.1 * 255 = 25.5` — environ 255 fois plus grand. Un pas
aussi grand peut faire "sauter" les poids loin au-delà de la valeur optimale à chaque exemple :
les poids oscillent au lieu de converger progressivement, voire divergent. Il peut aussi pousser
`z = w·x + b` vers des valeurs extrêmes qui saturent la sigmoïde, rendant le signal d'erreur peu
informatif.

Deux solutions existent :
1. **Réduire `lr`** en proportion inverse de l'échelle des `x_i` (ce que vous avez peut-être déjà
   fait en testant `champignons.csv` avec un `lr` plus petit). Ça fonctionne, mais il faut
   retrouver le bon `lr` pour chaque nouveau dataset.
2. **Normaliser les données** : ramener chaque caractéristique à une échelle commune (ex: `[0,1]`)
   avant l'entraînement, pour pouvoir garder le même `lr` quel que soit le dataset. C'est la
   pratique standard en apprentissage automatique, et c'est ce que vous allez implémenter à
   l'étape 7.

### Pourquoi mélanger (shuffle) les exemples à chaque époque ?

La règle de mise à jour ci-dessus ne mentionne pas l'ordre des exemples — on pourrait donc
croire que l'ordre n'a aucune importance. C'est une erreur fréquente, et elle vient d'une
confusion entre deux façons de faire de la descente de gradient :

- **Par lots (batch)** : on additionne les gradients de tous les exemples, puis on applique
  **une seule** mise à jour. L'addition étant commutative, l'ordre des exemples n'a alors
  strictement aucun effet sur le résultat.
- **En ligne (online / SGD)**, notre cas : `perceptron_train_step` est appelée **une fois par
  exemple**, et modifie immédiatement les poids. La mise à jour de l'exemple suivant utilise
  donc les poids *déjà modifiés* par l'exemple précédent. Comme la sigmoïde est non linéaire,
  appliquer `update(A)` puis `update(B)` ne donne en général **pas** le même résultat que
  `update(B)` puis `update(A)`. L'ordre influence donc la trajectoire des poids au fil de
  l'entraînement, même si la formule d'une seule étape ne le montre pas.

**Le risque concret** : si le fichier contient une longue série d'exemples de la même classe
à la suite, l'erreur `(label - y)` garde le même signe pendant toute la série : les poids sont
poussés dans une direction pendant plusieurs mises à jour consécutives, puis brutalement
corrigés au changement de classe. Si l'ordre ne change jamais d'une époque à l'autre, ce
phénomène se répète à l'identique à chaque passage.

**Mais attention** : l'ampleur de cet effet dépend fortement du dataset (longueur des séries
par classe, marge de séparation, `learning_rate`). Sur un petit dataset bien séparé, l'effet
peut être minime. C'est précisément ce que l'étape 9 vous demande de vérifier par
vous-mêmes plutôt que d'accepter l'argument tel quel.

Vous allez implémenter `dataset_shuffle()` (algorithme de Fisher-Yates) et l'appeler au début
de chaque époque dans `perceptron_train()` (étape 4).

## 4. Contexte théorique : évaluer une classification

L'accuracy (proportion d'exemples bien classés) est utile, mais insuffisante dès que les deux
types d'erreur n'ont pas le même coût, ou que les classes sont déséquilibrées. Pour analyser
plus finement un classifieur binaire, on utilise la **matrice de confusion**.

On choisit une classe dite **positive** et on classe chaque exemple selon quatre catégories :

|                             | Prédit positif | Prédit négatif |
|---|---|---|
| **Réellement positif** | Vrai Positif (TP) | Faux Négatif (FN) |
| **Réellement négatif** | Faux Positif (FP) | Vrai Négatif (TN) |

À partir de ces quatre compteurs, on définit :

- **Accuracy (justesse)** : `(TP + TN) / (TP + FP + TN + FN)` — proportion globale de bonnes
  prédictions, toutes classes confondues.
- **Précision** : `TP / (TP + FP)` — parmi les exemples prédits positifs, quelle proportion
  l'est réellement ? (mesure la fiabilité d'une prédiction positive)
- **Rappel (recall)** : `TP / (TP + FN)` — parmi les exemples réellement positifs, quelle
  proportion a été détectée ? (mesure la capacité à ne rien laisser passer)
- **F1-score** : `2 * (Précision * Rappel) / (Précision + Rappel)` — moyenne harmonique des
  deux, utile pour résumer le compromis précision/rappel en une seule valeur.

**Pourquoi l'accuracy ne suffit pas** : si 95% des exemples d'un dataset sont de classe 0, un
modèle qui prédit toujours 0 obtient 95% d'accuracy sans avoir rien appris — et un rappel nul
sur la classe 1. Le choix de la métrique à privilégier dépend du **coût relatif des deux types
d'erreur** dans le problème métier. C'est précisément l'enjeu du cas d'usage de la section 6 :
un faux négatif peut y être bien plus grave qu'un faux positif.

## 5. Architecture du projet fourni

```
.
├── Makefile
├── sujet.md
├── include/
│   ├── dataset.h
│   └── perceptron.h
├── src/
│   ├── dataset.c       <- à compléter
│   ├── perceptron.c    <- à compléter (partiellement)
│   └── main.c          <- fourni, ne pas modifier
└── data/
    └── champignons.csv
```

Compilez avec :
```
make
```
puis lancez par exemple :
```
./perceptron data/et.csv 1000 0.1
```
(arguments : fichier de données, nombre d'époques, taux d'apprentissage — les deux derniers
sont optionnels, valeurs par défaut 1000 et 0.1).

Un 4ème argument optionnel `train_ratio` (entre 0 et 1, ex: 0.8) active un découpage
train/test : le programme affiche alors l'accuracy sur l'ensemble d'entraînement **et** sur
l'ensemble de test, ainsi que la matrice de confusion et les métriques associées. Un 5ème
argument optionnel `normalize` (0 ou 1, défaut 0) active la normalisation min-max (calculée
sur le train set, appliquée au train et au test) :
```
./perceptron data/champignons.csv 1000 0.1 0.8 1
```
Sans le 4ème argument (ou avec la valeur 1.0), le programme entraîne et évalue sur
l'intégralité du dataset, comme avant, sans matrice de confusion ni normalisation.

## 6. Format des fichiers de données

Chaque ligne d'un fichier `.csv` représente un exemple :
```
x1,x2,...,xn,label
```
où `label` vaut 0 ou 1. Le nombre de colonnes de caractéristiques (`dim`) se déduit du fichier
lui-même (toutes les lignes ont le même nombre de colonnes).

`data/champignons.csv` a 4 caractéristiques `x1,x2,x3,x4`, valeurs entières entre 0 et 255 ;



## 7. Travail à réaliser

Le code est découpé en étapes. Respectez l'ordre : chaque étape dépend de la précédente.

### Étape 0 — Organisation du projet

Mettez en place l'arborescence du projet :
- un dossier `include/` pour les fichiers d'en-tête, un dossier `src/` pour les sources, le
  dossier `data/` contenant les jeux de données ;
- deux modules distincts : un module **dataset** (tout ce qui concerne les données : lecture,
  mélange, découpage, normalisation) et un module **perceptron** (le modèle, son apprentissage
  et son évaluation), chacun avec son en-tête et son fichier source ;
- un fichier source pour le programme principal ;
- un `Makefile` qui compile le tout en un exécutable `perceptron`, avec les options
  `-Wall -Wextra` (le projet doit compiler **sans aucun warning**), et une cible `clean`.

Protégez vos en-têtes contre les inclusions multiples et documentez chaque fonction publique
par un commentaire (rôle, paramètres, valeur de retour, qui est responsable de libérer la
mémoire).

### Étape 1 — Structures de données et gestion mémoire

Concevez les deux structures centrales du projet :
- un **dataset**, qui doit mémoriser le nombre d'exemples `n`, le nombre de caractéristiques
  `dim`, les caractéristiques de chaque exemple (un tableau de `n` lignes, chacune contenant
  `dim` réels) et le label (0 ou 1) de chaque exemple ;
- un **perceptron**, qui doit mémoriser son vecteur de poids (taille `dim`), son biais, sa
  dimension d'entrée et son taux d'apprentissage.

Écrivez, pour chacune de ces structures, une fonction de **création** (allocation dynamique
et initialisation — poids et biais à 0, ou petites valeurs aléatoires) et une fonction de
**libération** qui rend toute la mémoire allouée, dans le bon ordre (contenu d'abord, structure
ensuite). La création doit signaler un échec d'allocation (par exemple en retournant `NULL`)
sans laisser de fuite si l'échec survient en cours de route.


### Étape 2 — Lecture de fichier

Écrivez une fonction qui charge un dataset depuis un fichier CSV au format décrit en section 6
et retourne un dataset alloué dynamiquement, ou une valeur d'erreur si le fichier est
introuvable ou mal formé. Vous devez :
1. ouvrir le fichier et gérer l'échec d'ouverture ;
2. déterminer `dim` à partir de la première ligne (nombre de colonnes moins une pour le label) ;
3. déterminer `n` (nombre de lignes non vides) — par exemple par une première lecture
   complète, puis un retour au début du fichier ;
4. allouer la structure et ses tableaux ;
5. relire le fichier pour remplir caractéristiques et labels, en vérifiant que chaque ligne
   a bien le nombre de colonnes attendu ;
6. fermer le fichier, et libérer tout ce qui a été alloué en cas d'erreur.

*Testez cette étape isolément* : écrivez un petit programme de test qui charge `data/et.csv`
et affiche son contenu, puis vérifiez avec `valgrind` qu'aucune mémoire n'est perdue.

### Étape 3 — Prédiction

Écrivez :
- la fonction d'activation sigmoïde `σ(z) = 1 / (1 + e^(-z))` ;
- une fonction de prédiction qui, pour un perceptron et un vecteur d'entrée `x`, calcule
  `z = w·x + b` puis retourne `σ(z)`, valeur comprise dans ]0, 1[.

### Étape 4 — Apprentissage

Écrivez :
- une fonction qui effectue **une seule** mise à jour des poids et du biais pour un exemple
  `(x, label)`, selon la règle de la section 3 :
  `wᵢ ← wᵢ + lr · (label − y) · xᵢ` et `b ← b + lr · (label − y)`, avec `y` la prédiction
  courante ;
- une fonction qui mélange l'ordre des exemples d'un dataset **en place**, avec l'algorithme
  de Fisher-Yates (voir « Pourquoi mélanger... » en section 3). Indice : il suffit d'échanger
  les pointeurs de lignes et les labels, sans recopier les réels ;
- une fonction d'entraînement qui répète l'apprentissage pendant un nombre d'époques donné.
  Une époque = un passage complet sur tous les exemples. **Chaque époque doit commencer par un
  mélange du dataset**, avant de parcourir les exemples.

Pensez à initialiser le générateur aléatoire une seule fois, dans le programme principal.

### Étape 5 — Évaluation

Écrivez une fonction qui calcule l'**accuracy** d'un perceptron sur un dataset : la proportion
d'exemples bien classés, sous la forme d'un réel entre 0.0 et 1.0. Un exemple est classé 1 si
la prédiction est ≥ 0.5, et 0 sinon.

### Étape 6 — Programme principal

Écrivez le programme principal, qui prend en ligne de commande :
1. le fichier de données (obligatoire) ;
2. le nombre d'époques (optionnel, défaut 1000) ;
3. le taux d'apprentissage (optionnel, défaut 0.1) ;
4. le ratio d'entraînement `train_ratio` (optionnel, défaut 1.0 = pas de découpage) ;
5. l'option de normalisation (optionnel, 0 ou 1, défaut 0).

Sans argument, il affiche un message d'usage et se termine avec un code d'erreur. Pour
l'instant, il doit : charger le dataset, afficher `n` et `dim`, créer le perceptron, afficher
l'accuracy avant apprentissage, entraîner, afficher l'accuracy après apprentissage, afficher
les poids appris et le biais, puis tout libérer. Les arguments 4 et 5 seront exploités aux
étapes suivantes.

### Étape 7 — Séparation train/test

Jusqu'ici, vous avez entraîné et évalué le perceptron sur le **même** jeu de données : son
accuracy ne mesure donc que sa capacité à apprendre les exemples qu'il a déjà vus, pas sa
capacité à généraliser à des exemples nouveaux.

Écrivez une fonction qui sépare un dataset complet en deux datasets **indépendants**, `train`
et `test`, selon un ratio donné (ex : 0.8 = 80 % en entraînement, 20 % en test) :
- les `⌊n × ratio⌋` premiers exemples vont dans `train`, les suivants dans `test` ;
- le découpage se fait **dans l'ordre du fichier**, sans mélange ;
- les deux datasets produits ne partagent aucune mémoire avec le dataset d'origine et se libèrent séparément.

### Étape 8 — Normalisation des données

Comme expliqué en section 3 (« Pourquoi normaliser les entrées ? »), un taux d'apprentissage
qui fonctionne bien sur des données à petite échelle (`et.csv`, valeurs 0/1) peut faire
diverger ou osciller l'apprentissage sur des données à grande échelle (`champignons.csv`,
valeurs 0-255).

Concevez une structure de **paramètres de normalisation** contenant, pour chaque
caractéristique `j`, son minimum et son maximum. Écrivez :
- une fonction qui calcule ces paramètres sur un dataset — qui sera **toujours le train set,
  jamais le test set** (sinon, fuite d'information du test vers l'entraînement) ;
- une fonction qui applique la normalisation min-max `(x − min) / (max − min)` à chaque valeur
  d'un dataset, **en place**. Si `max == min` pour une caractéristique, évitez la division par
  zéro (convention : mettre 0.0). Appliquée au test set, la normalisation peut produire des
  valeurs hors de [0, 1] : c'est normal, ce n'est pas une erreur ;
- une fonction qui libère la mémoire de ces paramètres.

Intégrez-les au programme principal : si la normalisation est activée, calculez les paramètres
sur le train set, puis appliquez-les au train set et (s'il existe) au test set.

Vérifiez l'effet par vous-mêmes : lancez `champignons.csv` avec un taux d'apprentissage de 0.1,
une fois sans normalisation et une fois avec, et comparez l'accuracy de test.

### Étape 9 — Matrice de confusion et métriques

Concevez une structure de **matrice de confusion** contenant les quatre compteurs TP, FP, TN,
FN. Écrivez :
- une fonction qui parcourt un dataset, prédit chaque exemple et remplit la matrice de
  confusion par rapport à une **classe positive** passée en paramètre (0 ou 1) ;
- quatre fonctions qui calculent, à partir d'une matrice de confusion, l'accuracy, la
  précision, le rappel et le F1-score (définitions en section 4).

Gérez les divisions par zéro : si le dénominateur est nul (ex : `TP + FP == 0` pour la
précision), la convention retenue est de retourner 0.0.

Intégrez-les au programme principal : lorsqu'un split est actif, affichez la matrice de
confusion et les quatre métriques sur le test set.

### Étape 10 — Expérimentation et analyse (à inclure dans le rapport)

Testez votre perceptron sur les jeux de données fournis :

1. `data/et.csv` et `data/ou.csv` (sans split, sans normalisation) : le perceptron doit
   converger vers 100 % d'accuracy.

2. `data/champignons.csv` (le cas d'usage de la section 2), **avec split train/test et
   normalisation** :
   - comparez la convergence avec et sans normalisation pour un même taux d'apprentissage ;
   - calculez la matrice de confusion et les 4 métriques sur l'ensemble de test ;
   - justifiez votre choix de classe positive et la métrique à privilégier pour ce cas
     d'usage ;
   - concluez : peut-on faire confiance à votre perceptron pour remplacer le vieux sage ?

3. **Mesurez l'effet du shuffle** : désactivez temporairement le mélange en début d'époque,
   puis comparez l'accuracy obtenue après un petit nombre d'époques (ex : 5 à 10) avec et sans
   shuffle, sur `champignons.csv`. Documentez ce que vous observez, même si l'effet est faible,
   et proposez une explication

## 8. Rendu attendu

- Le code source complété (dossiers `src/` et `include/`).
- Un rapport court (1 à 2 pages, format libre — Markdown ou PDF) contenant :
  - un résumé de votre implémentation (choix faits, difficultés rencontrées) ;
  - les résultats obtenus sur les 5 datasets (accuracy finale, nombre d'époques utilisé,
    accuracy train vs test pour points2d et champignons) ;
  - la comparaison de convergence sur `champignons.csv` avec/sans normalisation, à
    `learning_rate` égal ;
  - la comparaison de convergence avec/sans shuffle, et votre explication de l'ampleur
    (ou de la faiblesse) de l'effet observé ;
  - la matrice de confusion et les métriques (précision, rappel, F1) sur `champignons.csv`,
    avec votre justification du choix de classe positive et de métrique prioritaire ;
  

## 9. Conseils

- Compilez et testez après chaque étape, pas seulement à la fin.
- Utilisez `valgrind` régulièrement, pas seulement avant le rendu.
- Pour déboguer la convergence, ajoutez temporairement un affichage de l'accuracy toutes les
  100 époques dans votre boucle d'apprentissage.
- N'hésitez pas à essayer plusieurs valeurs de `learning_rate` (par exemple 0.01, 0.1, 0.5) et
  à observer leur effet sur la vitesse de convergence. Sur `champignons.csv` (valeurs jusqu'à
  255), deux options pour rester stable : réduire fortement `learning_rate` (ex: 0.0001) sans
  normaliser, ou normaliser les données (`normalize = 1`) et garder un `learning_rate` usuel
  (ex: 0.1) — comparez les deux pour bien comprendre l'effet de la normalisation.
