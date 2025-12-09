# Algorithmique - Robo-courier
## Objectifs du Projet

L’objectif de ce projet est modélisé une ville sous forme de graphe pondéré. Puis de développer un programme Python permettant à un robo-coursier de réaliser une tournée de livraison de manière optimisée en utilisant 2 algorithmes différents : Dijkstra et A*.


## Fonctionnalités Clés
- **Planification multi-colis** : Choix de plusieurs lieux de livraison
- **Ordonancement des lieux de livraison** : Les lieux de livraison sont trié via une méthode gloutonne
- **Calcul du plus court chemin** : Grâce à l'algorithme de Dijkstra ou A*
- **Fonction optionnelle - Capacité de stockage limité** : Le robot peut transporter une quantité limitée de colis
- **Fonction optionnelle - Arrêtes à risque** : des évenment peuvent blouquer certains chemins de manière aléatoire.


## Dépendances et bibliothèques
- **Python** : Langage de programmation
- **Networkx** : pour la gestion des graphes
- **Matplotlib** : pour tracer le graphique
- **Heapq** : pour la file de priorité
- **math** : pour le calcul des distances
- **random** : pour la création de nombre aléatoire
- **time** : pour mesurer le temps d'exécution

  

## Utilisation
1. Lancer le programme 
2. Choisir les lieux de livraison parmi la liste proposé (écrire le nom de lieux exactement comme indiqué dans la liste)
3. Choisir d'inclure les fonction optionnelles (o/n)
4. Choisir les algorithmes à utiliser :
    - Dijkstra
    - a_star + heuristique 
    - both (comparaison entre Dijkstra et A*)
5. Une fenêtre s'ouvre avec la carte et l'animation du trajet du robot
6. Pour quitter le programme il suffit de fermer la fenêtre
   

## Contributeurs
- Raphaël Maul
- Alexandre Raffin

