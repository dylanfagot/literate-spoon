# -*- coding: utf-8 -*-
"""
Ce fichier permet de lancer l'analyse des CSV contenus dans un dossier
"donnees_cours_CAC40" contenu a la racine du projet, sur la base de fonctions
contenues dans le fichier fonctions.py a la racine du projet
L'analyse est basee sur la publication

Ganeshapillai, G., Guttag, J., & Lo, A. (2013, May). 
Learning connections in financial time series. 
In International Conference on Machine Learning (pp. 109-117). PMLR.

@auteur: Dylan Fagot
"""

import numpy as np
import fonctions as f

if __name__ == "__main__":
    
    """
    Parametres initiaux
    """
    alpha = 0.5 # ponderation du risque
    lbda = 1e-4 # regularisation de P
    taille_historique = 100 # nombre de jours pris en compte dans les donnees
    
    """
    --------- Chargement des donnees et calcul des rendements ------
    """
    # Dossier contenant les CSV a lire
    chemin_csv = "donnees_cours_CAC40"
    
    # On appelle la fonction de lecture des valeurs de cours
    donnees_lues, liste_noms_cours = f.lecture_donnees(chemin_csv)
    
    # Calcul des rendements associes a chaque cours sur les
    # taille_historique dernières valeurs de cours
    rendements = f.calculer_rendement(donnees_lues)[-taille_historique::]
    
    # Rendements moyens par cours
    rendements_moyens = np.mean(rendements, axis=0)
    
    """
    --------- Analyse sur matrice de covariance ------
    """
    
    print(" --------- Calcul sur covariances... ---------")
    
    # Calcul de la matrice de covariances
    matrice_Q = np.cov(rendements, rowvar=False)
    
    # Resolution du probleme de diversification via covariances
    matrice_Q_sans_CAC40 = matrice_Q[:-1, :-1]
    solution_Q, rendement_cible_Q = f.optimiser_diversite(matrice_Q_sans_CAC40, rendements_moyens[:-1], alpha)
    w_Q = solution_Q.x
    
    """
    --------- Analyse sur matrice de connexions ------
    """
    
    print(" --------- Calcul sur connexions... ---------")
    # Apprentissage de la matrice de connexions
    a_opt, P_opt, matrice_C = f.apprentissage_connexions(matrice_Q, rendements, lbda)

    # Resolution du probleme de diversification via connexions
    matrice_C_sans_CAC40 = matrice_C[:-1, :-1]
    solution_C, rendement_cible_C = f.optimiser_diversite(matrice_C_sans_CAC40, a_opt, alpha)
    w_C = solution_C.x

    """
    --------- Enregistrement des donnees d'initialisation ---------
    """
    
    print(" --------- Enregistrement des parametres... ---------")
    np.savez("parametres_initiaux.npz", a_opt = a_opt, matrice_C = matrice_C, 
             matrice_Q = matrice_Q, w_Q = w_Q, w_C = w_C,
             rendements = rendements, rendements_moyens = rendements_moyens)
    
    
    
    
    
    
    
    
    
    
    
    