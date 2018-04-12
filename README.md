# The-MNIST-dataset
deep_learning_exo1

"""
Activité #1: Visualiser • Créer un ﬁchier “plot_mnist_images_exo1.py”;
• Observer deep_learning_exo1.py et charger le MNIST avec Keras dans le programme “plot_mnist_images_exo1.py”;
• Réutiliser le “plot_gallery” de la formation T-SNE pour visualiser le dataset MNIST. Par exemple, 8 lignes et 18 colonnes de chiﬀres.
C’est le type d’outils qui sert tout le temps. A mettre de coté.
deep_learning_exo1:  The MNIST dataset
Le bloc de base des réseaux de neurones est la "couche", un module de traitement de données que vous pouvez concevoir comme un "ﬁltre" pour les données. Certaines données arrivent et sortent sous une forme plus utile. Précisément, les couches extraient des représentations à partir des données qui y sont introduites - avec un peu de chance des représentations plus signiﬁcatives pour le problème en question. 
La plupart des apprentissages profonds consistent en fait à enchaîner des couches simples qui implémenteront une forme de «distillation de données» progressive. Un modèle d'apprentissage en profondeur est comme un tamis pour le traitement des données, constitué d'une succession de ﬁltres de données de plus en plus raﬃnés - les «couches»(voir explications données à l’oral).
Ici, notre réseau se compose d'une séquence de deux couches “Dense”, qui sont des couches de neurones densément connectées (également appelées «entièrement connectées»). La deuxième (et dernière) couche est une couche «softmax» à 10 voies, ce qui signiﬁe qu'elle retournera un tableau de 10 scores de probabilité (sommation à 1). Chaque score sera la probabilité que l'image digitale actuelle appartienne à l'une de nos classes à 10 chiﬀres.
Pour préparer notre réseau à la formation, nous devons choisir trois autres éléments, dans le cadre de l'étape de «compilation»:
• Une fonction de perte (Loss function): comment le réseau pourra-t-il mesurer la qualité de son travail sur ses données d'entraînement, et comment il sera capable de se diriger dans la bonne direction?
• Un optimiseur (Optimizer): c'est le mécanisme par lequel le réseau se mettra à jour en fonction des données qu'il voit et de sa fonction de perte.
• Mesures (Metrics) à surveiller pendant l'entraînement et les tests. Ici, nous ne nous intéresserons qu'à la précision (la fraction des images correctement classées).
Crédit : François Chollet, Keras, livre “Deep Learning”, éditions Manning
deep_learning_exo1:  The MNIST dataset
Activité #2: Rechercher dans un navigateur la justiﬁcation des choix du programme par rapport à la structure du dataset: • Keras optimizer rmsprop
• Keras loss=‘categorical_crossentropy'
• keras metrics=['accuracy']
deep_learning_exo1:  The MNIST dataset Activité #2: Solutions • Keras optimizer rmsprop : liens utiles 
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

• Keras loss=‘categorical_crossentropy'
https://keras.io/losses/
Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample). In order to convert integer targets into categorical targets, you can use the Keras utility to_categorical:
• keras metrics=['accuracy']
https://keras.io/metrics/
deep_learning_exo1:  The MNIST dataset Activité #3: Compléter le “reshaping” and “scaling” # on veut que train_images soit un array de 60000 par 784 (Pourquoi 784?)
train_images = train_images     à modiﬁer
# train_images, ce sont des nombres entiers (0->255), on veut les transformer en nombres réels puis passer de 0->1
train_images = train_images     à modiﬁer
# même traitement avec test_images
test_images = test_images     à modiﬁer
test_images = test_images      à modiﬁer A retenir : comment en Python, on change le “typage” d’une variable
deep_learning_exo1:  The MNIST dataset Activité #4: Utiliser Keras pour encoder “categorical” from keras.utils import à compléter
print(train_labels.shape)
train_labels =  à compléter   (train_labels)
print(train_labels.shape)
test_labels =   à compléter  (test_labels)
A retenir: comment travailler avec des labels “categorical”
deep_learning_exo1:  The MNIST dataset
Le code: network.ﬁt(train_images, train_labels, epochs=5, batch_size=128) test_loss, test_acc = network.evaluate(test_images, test_labels) print('test_acc:', test_acc) • Sélectionner un lot d'exemples d'entraînement x et la cible correspondante y;
• exécuter le “network” sur x (une étape appelée la passe vers l’avant) pour obtenir des prédictions y_pred;
• calculer la perte du “network” sur le lot, mesurer la discordance entre y_pred et y;
• mettre à jour tous les poids du “network” d'une manière qui réduit légèrement la perte sur ce lot;
• calculer le gradient de la perte par rapport aux paramètres du “network” (une passe vers l’arrière);
• déplacer les paramètres un peu en sens inverse du gradient, réduisant ainsi un peu la perte sur le lot.
Mini-batch stochastic gradient descent or “min-batch SGD”
deep_learning_exo1:  The MNIST dataset La notion de “momentum” : Le “momentum” tire son inspiration de la physique. 
Une image mentale utile ici consiste à considérer le processus d'optimisation comme une petite boule roulant sur la courbe des pertes. Si elle a assez d'élan, la balle ne sera pas bloquée dans un ravin et ﬁnira au minimum global. L'impulsion est mise en œuvre en déplaçant la balle à chaque pas en fonction non seulement de la valeur actuelle de la pente (accélération actuelle), mais aussi de la vitesse actuelle (résultant de l'accélération précédente).
Crédit : François Chollet, Keras, livre “Deep Learning”, éditions Manning

deep_learning_exo1:  The MNIST dataset En conclusion : Vous avez pu revoir plusieurs concepts déjà abordés auparavant:
• La “vectorization” et la normalisation des données ;
• Le “One-Hot-Encoder” (version Scikit-Learn) qui facilite l’apprentissage du deep learning.
Vous venez de voir comment nous pourrions construire et former un réseau de neurones pour classer les chiﬀres manuscrits, en moins de 20 lignes de code Python. 
Dans le chapitre suivant, nous allons détailler chaque élément que nous venons de voir, et clariﬁer ce qui se passe réellement dans les coulisses. Vous apprendrez à connaître les «tenseurs», les objets stockant des données entrant dans le réseau, les opérations de tenseurs, les couches constituées et la descente de gradient, ce qui permet à notre réseau d'apprendre de ses exemples d’entraînement.
Crédit : François Chollet, Keras, livre “Deep Learning”, éditions Manning
"""