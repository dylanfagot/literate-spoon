# -*- coding: utf-8 -*-
"""
Ce fichier contient l'ensemble des fonctions utilises par le script
script_analyse_cours.py

@auteur: Dylan Fagot
"""

import numpy as np
import matplotlib.pyplot as plt
import os, csv
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
import itertools

# On supprime toutes les fenetres matplotlib eventuellement ouvertes
plt.close("all")

def calcul_ponderation_top_k(retour):
    """
    Cette fonction permet d'accentuer le poids des retours autours de -10%
    f(r_t,k) dans la publication

    Parameters
    ----------
    retour : tableau NumPy

    Returns
    -------
    ponderation_retour : tableau NumPy

    """
    
    ponderation_retour = np.exp(-(retour + 0.1)**2)
    
    return ponderation_retour


def lecture_donnees(chemin_dossier):
    """
    Cette fonction parcourt le dossier en argument pour en extraire le 
    contenu des fichiers csv presents

    Parameters
    ----------
    chemin_dossier : str, chemin vers le dossier contenant les CSV de donnees

    Returns
    -------
    dataframe_global : dataFrame valeurs dans le temps (lignes) pour chaque
    cours extrait (colonnes)

    """
    
    # On liste l'ensemble des fichiers (CSV ou non) dans le dossier cible
    liste_fichiers_cours = os.listdir(chemin_dossier)
    
    # Listes vides pour conserver les noms des cours et realiser la fusion
    # des donnes lues sous forme de dataFram
    liste_noms_cours = []
    liste_dataframe_cours = []
    
    # Pour chaque fichier dans le dossier
    for fichier in liste_fichiers_cours:
        # On recupere le nom
        nom_cours = fichier.split('.')[0]
        # Puis l'extension
        extension = fichier.split('.')[1]
        
        # Si c'est un fichier CSV
        if (extension == "txt"):
            # Lecture de celui-ci via fonction lecture_csv()
            dataframe_cours = lecture_csv(chemin_dossier, fichier) # os.merge
        
            # On enregistre le nom du fichier et conserve ses donnees
            liste_noms_cours.append(nom_cours)
            liste_dataframe_cours.append(dataframe_cours)
    
    # Construction iterative du dataFrame global depuis le premier dataFrame
    dataframe_global = liste_dataframe_cours[0]
    
    # On fusionne chaque dataFrame au dataFrame global
    for i in range(len(liste_dataframe_cours)):
        dataframe_global = dataframe_global.merge(liste_dataframe_cours[i])
    
    # Sauvegarde du dataFrame global dans fichier .npy
    np.save("dataframe.npy", dataframe_global)
    
    print("Extraction des cours du dossier " + chemin_dossier + " terminee")
    return dataframe_global


def lecture_csv(chemin_dossier, fichier):
    """
    Cette fonction lit le fichier csv donne par le chemin et le nom du fichier
    et le convertit en objet Pandas

    Parameters
    ----------
    chemin_dossier : chemin vers les fichiers CSV
    fichier : nom du fichier csv

    Returns
    -------
    series_cours : donnes du csv sous forme d'objet dataFrame

    """
    
    with open(chemin_dossier  + "\\" + fichier, newline = "")  as fichier_csv:
        
        # Extraction du contenu du CSV
        contenu_csv = csv.reader(fichier_csv, delimiter="\t")
        
        # On prepare des listes pour associer dates/valeurs lues
        liste_dates = []
        liste_valeurs = []
        
        # On parcourt le fichier pour recuperer les dates et valeurs de cloture
        # dans le dictionnaire cree
        for (i_ligne, ligne) in enumerate(contenu_csv):
            if (i_ligne == 0):
                # Lecture de la premiere ligne du fichier (entetes)
                # print("Contenu fichier : " + str(ligne[:-1]))
                print("Lecture du fichier {}...".format(fichier))
            else:
                # Lecture de donnes numeriques, on les sauvegarde
                (date, ouv, haut, bas, clot, vol, devise, vide) = ligne
                # Recuperation des dates et valeurs de clotures associees
                liste_dates.append(date)
                liste_valeurs.append(float(clot))
    
    # Construction d'un dictionnaire en dates (clés) et valeurs de cloture (valeurs)
    d = {"dates" : liste_dates, "valeurs_"+fichier : liste_valeurs}
    
    # Transformation du dictionnaire en objet dataFrame
    dataFrame_cours = pd.DataFrame(d)
    
    return dataFrame_cours


def calculer_rendement(dataFrameGlobal):
    """
    Calcule le rendement de chaque cours et enregistre le resultat
    sous forme d'une tableau NumPy

    Parameters
    ----------
    dataFrameGlobal : dataFrame contenant les valeurs des differents cours

    Returns
    -------
    rendements : tableau NumPy de taille (T x C)

    """
    
    # Conversion du dataFrame sans dates en tableau NumPy de reels
    tableau_numpy = ((dataFrameGlobal.to_numpy())[:, 1:]).astype(float)
    
    # On recupere le nombre de cours du tableau resultant
    n_cours = tableau_numpy.shape[1]
    
    # On calcule les rendements, avec r=0 pour t=0
    rendements = np.concatenate((np.zeros((1, n_cours)), np.diff( tableau_numpy, 1, axis=0))) / tableau_numpy
    
    #rendements_test = rendements[:100,:]
    
    return rendements


def eigh_decroissant(matrice):
    """
    Cette fonction permet de realiser la decomposition en valeurs propres
    d'une matrice en triant les valeurs propres par ordre decroissant

    Parameters
    ----------
    matrice : matrice semi-definie positive de taille N x N

    Returns
    -------
    valeurs_propres : N valeurs propres triees par ordre decroissant
    vecteurs_propres : N vecteurs propres

    """
    
    # Tout d'abord, on realise la diagonalisation
    valeurs_propres, vecteurs_propres = np.linalg.eigh(matrice)
    
    # On recupere les indices des valeurs propres triees (ordre decroissant)
    ordre_decroissant = np.argsort(-valeurs_propres)
    
    # Les indices servent ensuite a trier simultanement valeurs et vecteurs
    valeurs_propres_triees = valeurs_propres[ordre_decroissant]
    vecteurs_propres_tries = vecteurs_propres[:, ordre_decroissant]
    
    return valeurs_propres_triees, vecteurs_propres_tries


def optimiser_diversite(matrice, rendements_moyens, alpha):
    """
    Fonction de resolution du probleme de maximisation de diversite
    sous contrainte de rendement, norme 1 et positivite des poids

    Parameters
    ----------
    matrice : matrice de covariances ou de connexions
    
    rendements_moyens : vecteur contenant les rendements moyens de chaque cours
    
    alpha : parametre de compromis de risque sur rendement/diversite
    (0 <= alpha <= 1)

    Returns
    -------
    solution du probleme d'optimisation sous forme d'objet SciPy
    OptimizeResult, le vecteur w solution est donnee par son attribut x

    """
    
    def evaluer(w):
        # Fonction permettant d'evaluer la covariance totale a partir
        # du vecteur de poids w : evaluer(w) = w' x Q x w
        covariance_totale = np.dot(w.transpose(), np.dot(matrice, w))
        return covariance_totale
    
    print("Appel de optimiser_diversite()")
    
    # Initialisation du vecteur de poids : poids uniformes
    n_cours = matrice.shape[0]
    w_0 = np.ones(n_cours) / n_cours
    
    # Contraintes de norme 1 et de positivite des poids
    contrainte_norme_un = LinearConstraint(np.ones(n_cours), [1], [1])
    bounds = Bounds(np.zeros(n_cours), np.ones(n_cours))
    
    # Contraintes de rendement
    rendement_cible = alpha * np.max(rendements_moyens)
    contrainte_rendement = LinearConstraint(rendements_moyens, [rendement_cible], [np.inf])

    # Resolution du probleme de maximisation de diversite sous contrainte de
    # rendement et de norme du vecteur des poids
    solution = minimize(evaluer, w_0, method='trust-constr',
                   constraints=[contrainte_norme_un, contrainte_rendement],
                   options={'verbose': 1}, bounds=bounds)
    
    return solution, rendement_cible


def calcul_retours_modelises(a, P, *args):
    """
    Cette fonction permet de calculer les rendements modelises sur la base des
    parametres obtenus lors de l'apprentissage des connexions

    Parameters
    ----------
    a : vecteur contenant les parametres a_c (taille C)
    P : matrice permettant de calculer les connexions sous la forme P'P
    (taille (C+1)x(C+1))
    
    *args : arguments incluant les rendements empiriques

    Returns
    -------
    retours_modelises : tableau NumPy de meme taille que celui des rendements
    empiriques (taille T x C)

    """
    
    # On recupere les retours empiriques et les tailles
    retours = args[0][0]
    T = retours.shape[0]
    n_cours = retours.shape[1]-1
    
    # Initialisation du calcul des retours modelises et des termes d
    retours_modelises = np.zeros((T, n_cours))
    d = np.zeros((T, n_cours))
    
    # Pour chaque instant t, pour chaque cours k
    for t, k in itertools.product(range(T), range(n_cours)):
        
        # Calcul du terme d
        d[t, k] = a[k] + retours[t,-1]*np.dot(P[k,:-1], P[-1,:-1])
        
        # Calcul en plusieurs temps du dernier terme basé sur les connexions
        terme_retour_sans_j_k = retours[t, -1] - d[t, :]
        terme_retour_sans_j_k[k] = 0
        somme_j = np.dot(terme_retour_sans_j_k, P[:-1, :-1])
        
        # Calcul final du retour modelise (t,k)
        retours_modelises[t,k] =  d[t, k] + np.dot(P[k, :-1], somme_j)
    
        
    return retours_modelises


def objectif(aP, *args):
    """
    Fonction permettant de calculer la fonction objectif d'apprentissage
    pour un vecteur (a,P) donne

    Parameters
    ----------
    aP : vecteur contenant les a et P (taille C(C+1))
    *args : arguments pour la fonction contenant les retours et lambda

    Returns
    -------
    valeur_objectif : valeur de la fonction objectif au vecteur aP donne

    """
    
    # On recupere le nombre de cours depuis la 
    # taille de aP = n_cours^2 + 3*n_cours + 1
    n_cours = int( (-3+np.sqrt(9-4*(1-aP.size)))/2 + 1 ) - 1
    
    # On decoupe le vecteur aP en a de taille C, et en P de taille CxC
    a = aP[:n_cours]
    P = np.reshape(aP[n_cours:], (n_cours+1, n_cours+1))
    
    # Calcul des retours modelises pour les valeurs de a et P donnees
    retours_modelises = calcul_retours_modelises(a, P, args)
    
    # On recupere les arguments separement
    retours = args[0]
    lambd = args[1]
    
    # Initialisation du tableau permettant de sommer les residus
    T = retours.shape[0]
    valeurs = np.zeros((T, n_cours))
    
    # Pour chaque residus (t, k)
    for t, k in itertools.product(range(T), range(n_cours)):
        valeurs[t, k] = calcul_ponderation_top_k(retours[t, k]) * (retours[t, k] - retours_modelises[t, k])**2
    
    # On somme tous les residus avec le terme de penalisation
    valeur_objectif = np.sum(valeurs) + lambd * ( np.sum(a**2) + np.sum(P**2))
    
    return valeur_objectif


def apprentissage_connexions(covariances, *args):
    """
    Cette fonction permet de realiser l'apprentissage des connexions
    sur la base de la matrice de covariances, des retours et des parametres

    Parameters
    ----------
    covariances : matrice de covariances de taille CxC
    *args : arguments pour la fonction contenant les retours et lambda

    Returns
    -------
    a_opt : resultat de l'optimisation sur les termes a
    P_opt : resultat de l'optimisation sur P
    connections : resultat de l'optimisation (matrice de connexions)
    solution : solution du probleme d'optimisation sous forme d'objet
    SciPy
    """
    
    print("Appel de apprentissage_connexions()")
    
    # Initialisation de a avec les rendements moyens
    n_cours = covariances.shape[0] - 1 # nombre de cours, hors CAC40
    a_0 = np.mean((args[0])[:, :-1], axis=0) # moyenne de retour k ~= a_k
    
    # Initialisation de la matrice P via decomposition de Cholesky 
    # de la matrice de covariance
    P_0 = np.linalg.cholesky(covariances) # P = racine(cov)
    
    # Creation du vecteur de parametres aP = [a, P]
    aP_0 = np.concatenate((a_0, P_0.flatten()))
    
    print("Cout initial = {}".format(objectif(aP_0, args[0], args[1])))
    
    # Appel de la methode d'optimisation
    solution = minimize(objectif, aP_0, args=args, method="BFGS",
                      options={'gtol': 1e-6, 'disp': True})
    
    # Extraction de la solution (a, P, connections)
    aP_opt = solution.x
    cout = solution.fun
    print("Cout final = {}".format(cout))
    
    a_opt = aP_opt[:n_cours]
    P_opt = np.reshape(aP_opt[n_cours:], (n_cours+1, n_cours+1))
    connection = np.dot(P_opt.transpose(), P_opt)
    
    return a_opt, P_opt, connection, solution

