# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:54:22 2023

@author: dylan
"""

import numpy as np
import fonctions as f
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    
    # Chargement du fichier de sauvegarde des parametres
    dictt = np.load("parametres_initiaux.npz")
    
    plt.close("all")
    
    # On recupere les parametres initiaux
    a_opt = dictt["a_opt"]
    matrice_Q = dictt["matrice_Q"]
    matrice_C = dictt["matrice_C"]
    w_Q = dictt["w_Q"]
    w_C = dictt["w_C"]
    rendements = dictt["rendements"]
    rendements_moyens = dictt["rendements_moyens"]
    
    # Decomposition en valeurs propres et calcul du conditionnement
    matrice_Q_sans_CAC40 = matrice_Q[:-1,:-1]
    val_Q, vec_Q = f.eigh_decroissant(matrice_Q_sans_CAC40)
    cond_Q = val_Q[0] / val_Q[-1]
    
    # Decomposition en valeurs propres et calcul du conditionnement
    matrice_C_sans_CAC40 = matrice_C[:-1,:-1]
    val_C, vec_C = f.eigh_decroissant(matrice_C_sans_CAC40)
    cond_C = val_C[0] / val_C[-1]
    
    """
    --------------- Precision de modelisation --------------------------
    """
    
    # Calcul des cours modelises avant (via P_0) et apres apprentissage
    # des connexions (via P)
    P_0 = np.linalg.cholesky(matrice_Q)
    P = np.linalg.cholesky(matrice_C)
    rendements_modelises_avant = f.calcul_retours_modelises(rendements_moyens, P_0, (rendements, 0))
    rendements_modelises_apres = f.calcul_retours_modelises(a_opt, P, ((rendements, 0)))
    
    # Affichage des cours reels / modelises pour le cours 1
    numero_cours = 1
    plt.figure(0)
    plt.subplot(3,1,1)
    plt.plot(rendements[:, numero_cours], 'b', label="valeurs réelles cours #{}".format(numero_cours))
    plt.plot(rendements_modelises_avant[:, numero_cours], 'r:', label="modelisation de départ")
    plt.plot(rendements_modelises_apres[:, numero_cours], 'r', label="modélisation cours #{}".format(numero_cours))
    plt.legend()
    plt.grid()
    plt.xlabel("Temps")
    plt.ylabel("Valeurs")
    
    # Affichage des cours reels / modelises pour le cours 27
    numero_cours = 27
    plt.subplot(3,1,2)
    plt.plot(rendements[:, numero_cours], 'b', label="valeurs réelles cours #{}".format(numero_cours))
    plt.plot(rendements_modelises_avant[:, numero_cours], 'r:', label="modelisation de départ")
    plt.plot(rendements_modelises_apres[:, numero_cours], 'r', label="modélisation cours #{}".format(numero_cours))
    plt.legend()
    plt.grid()
    plt.xlabel("Temps")
    plt.ylabel("Valeurs")
    
    # Affichage du cours reel CAC40, servant de base a la modelisation
    plt.subplot(3,1,3)
    plt.plot(rendements[:, -1], 'b', label="valeurs réelles CAC40")
    plt.legend()
    plt.grid()
    plt.xlabel("Temps")
    plt.ylabel("Valeurs")
    
    
    """
    ----------- Visualisation des matrices covariances / connexions ---------
    """
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(matrice_Q_sans_CAC40)
    plt.xlabel("Matrice de covariances")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(matrice_C_sans_CAC40)
    plt.xlabel("Matrice de connexions")
    plt.colorbar()
    
    
    """
    ----------- Analyse spectre des matrices covariances / connexions ---------
    """
    plt.figure(2)
    plt.semilogy(val_Q, 'b', label="matrice de covariances, cond={:5.2e}".format(cond_Q))
    plt.semilogy(val_C, 'r' , label="matrice de connexions, cond={:5.2e}".format(cond_C))
    plt.legend()
    plt.grid()
    plt.title("Valeurs propres (ordre décroissant)")
    
    
    """
    ----------- Proportions d'investissement sur cours  ---------
    """
    plt.figure(3)
    plt.plot(w_Q*100, "bx", label="via covariances, $r_e$ = {:5.2e}".format(np.dot(rendements_moyens[:-1], w_Q)))
    plt.plot(w_C*100, "ro", label="via connexions, $r_e$ = {:5.2e}".format(np.dot(a_opt, w_C)))
    plt.legend()
    plt.grid()
    plt.title("Proportions optimales")
    plt.xlabel("Cours")
    plt.ylabel("Proportion [%]")
    