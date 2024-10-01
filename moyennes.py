import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import cmath
import math
#################################################################################################   
#                                           FONCTION UTILES                                     #
#################################################################################################

def string_to_float(tableau):
    valeurs_converties = []

    for cellule in tableau:
        # Ignorer les chaînes vides
        if cellule.strip() != '':
            # Convertir en float avec 2 chiffres après la virgule
            valeur_convertie = round(float(cellule), 2)
            valeurs_converties.append(valeur_convertie)

    return valeurs_converties

def tri_croissant(tableau):
    n = len(tableau)

    for i in range(n - 1):
        # Trouver l'élément minimum dans le reste du tableau
        index_min = i
        for j in range(i + 1, n):
            if tableau[j] < tableau[index_min]:
                index_min = j

        # Échanger l'élément minimum avec l'élément à la position actuelle
        tableau[i], tableau[index_min] = tableau[index_min], tableau[i]
        
def tri_decroissant(tableau):
    
    n = len(tableau)

    for i in range(n - 1):
        # Trouver l'élément minimum dans le reste du tableau
        index_min = i
        for j in range(i + 1, n):
            if tableau[j] > tableau[index_min]:
                index_min = j

        # Échanger l'élément minimum avec l'élément à la position actuelle
        tableau[i], tableau[index_min] = tableau[index_min], tableau[i]  

def recuperer_donnees_fichier(fichier):
    # Lire les données du fichier CSV
    donnees = []
    with open(fichier, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            donnees.append(row)

    return donnees

def moyenne_impedance_fichier(fichier): # calculer la moyenne des magnitudes de toutes les mesures d'un fichier à chaque fréquence
    # Récupérer les données du fichier  
    donnees_fichier = recuperer_donnees_fichier(fichier)
    #print( "données fichier: ",donnees_fichier)
    # Vérifier si des données ont été récupérées
    if donnees_fichier is None:
        return None

    # Liste pour stocker les moyennes
    moyennes = []

    # Lignes à moyenner
    lignes_a_moyenner = []
    
    frequences=string_to_float(donnees_fichier[1])
    
    nb_lignes=len(donnees_fichier)
    nb_mesures=int(nb_lignes/4)
    nb_colonnes=len(donnees_fichier[1])
    #print("fichier : ", fichier)
    #print("nb mesures", nb_mesures)
    for i in range(nb_mesures):
        lignes_a_moyenner.append(2+4*i)
    #print('lignes a moyenner',lignes_a_moyenner)
    #print('nb lignes fichier :',nb_lignes)
    #print('frequences:', frequences)
    #print('nb pt freq (colonnes) : ', len(frequences))
    
    for colonne in range(len(frequences)):
        sum_imp=0
        for ligne in lignes_a_moyenner:
            sum_imp = sum_imp + round(float(donnees_fichier[ligne][colonne]),2)
            moy_col=round((sum_imp*4)/nb_lignes,1)
        moyennes.append(moy_col)
    
    #print('nb de moy :', len(moyennes))
    return moyennes, frequences

def impedance_a_1_freq(fichier, freq): # calculer la moyenne des magnitudes de toutes les mesures d'un fichier à chaque fréquence
    # Récupérer les données du fichier  
    donnees_fichier = recuperer_donnees_fichier(fichier)
    #print( "données fichier: ",donnees_fichier)
    # Vérifier si des données ont été récupérées
    if donnees_fichier is None:
        return None

    # Liste pour stocker les moyennes
    impedances = []

    # Lignes à moyenner
    lignes_a_moyenner = []
    
    frequences=string_to_float(donnees_fichier[1])
    indice_freq = min(range(len(frequences)), key=lambda i: abs(frequences[i] - freq))
    freq_value = frequences[indice_freq]
    nb_lignes=len(donnees_fichier)
    nb_mesures=int(nb_lignes/4)
    nb_colonnes=len(donnees_fichier[1])
    #print("fichier : ", fichier)
    #print("nb mesures", nb_mesures)
    for i in range(nb_mesures):
        lignes_a_moyenner.append(2+4*i)
    #print('lignes a moyenner',lignes_a_moyenner)
    #print('nb lignes fichier :',nb_lignes)
    #print('frequences:', frequences)
    #print('nb pt freq (colonnes) : ', len(frequences))
    
    for ligne in lignes_a_moyenner:
            imp =round(float(donnees_fichier[ligne][indice_freq]),2)
            #print(imp)
            impedances.append(imp)
    
    #print('nb de moy :', len(moyennes))
    return impedances, freq_value

def moyenne_phase_fichier(fichier): # calculer la moyenne des magnitudes de toutes les mesures à chaque fréquence
    # Récupérer les données du fichier
    donnees_fichier = recuperer_donnees_fichier(fichier)
    #print( "données fichier: ",donnees_fichier)
    # Vérifier si des données ont été récupérées
    if donnees_fichier is None:
        return None

    # Liste pour stocker les moyennes
    moyennes = []

    # Lignes à moyenner
    lignes_a_moyenner = []
    
    frequences=string_to_float(donnees_fichier[1])
    
    nb_lignes=len(donnees_fichier)
    nb_mesures=int(nb_lignes/4)
    nb_colonnes=len(donnees_fichier[1])
    print("nb mesures", nb_mesures)
    for i in range(nb_mesures):
        lignes_a_moyenner.append(3+4*i)
    print('lignes a moyenner',lignes_a_moyenner)
    #print('nb lignes fichier :',nb_lignes)
    #print('frequences:', frequences)
    #print('nb pt freq (colonnes) : ', len(frequences))
    
    for colonne in range(len(frequences)):
        sum_imp=0
        for ligne in lignes_a_moyenner:
            sum_imp = sum_imp + round(float(donnees_fichier[ligne][colonne]),2)
            moy_col=round((sum_imp*4)/nb_lignes,1)
        moyennes.append(moy_col)
    
    #print('nb de moy :', len(moyennes))
    return moyennes, frequences

#################################################################################################   
#                                           MOYENNES                                            #
#################################################################################################   
def moyennes_dossier(dossier):
    # Obtenir la liste des fichiers CSV dans le dossier
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

    # Vérifier s'il y a au moins un fichier CSV dans le dossier
    if not fichiers_csv:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None

    # Créer une figure avec deux sous-graphiques (pour l'impédance et la phase)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Parcourir tous les fichiers CSV dans le dossier
    for fichier in fichiers_csv:
        chemin_fichier = os.path.join(dossier, fichier)

        # Calculer les moyennes pour le fichier actuel (impédance et phase)
        moy_impedance, freq_impedance = moyenne_impedance_fichier(chemin_fichier)
        moy_phase, freq_phase = moyenne_phase_fichier(chemin_fichier)

        # Tracer l'impédance sur le premier sous-graphique
        axes[0].plot(freq_impedance, moy_impedance, linestyle='-', label=fichier)

        # Tracer la phase sur le deuxième sous-graphique
        axes[1].plot(freq_phase, moy_phase, linestyle='-', label=fichier)

    # Configuration des axes et affichage des légendes
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Fréquence (Hz)')
    axes[0].set_ylabel('Impédance magnitude ')
    axes[0].grid(True)
    axes[0].set_ylim(0, 2000)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('Fréquence (Hz)')
    axes[1].set_ylabel('Impédance phase ')
    axes[1].grid(True)

    # Ajuster la disposition pour un meilleur espacement
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
    
def print_moyennes_dossier(dossier): # rempli un tableau contenant les lignes de moyennes
  
 # Liste des fichiers CSV dans le dossier
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

    # Vérifier s'il y a au moins un fichier CSV dans le dossier
    if not fichiers_csv:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None

    # Tableau 2D pour stocker les moyennes de chaque fichier
    moyennes_dossier = [[]]
    moyenne_20khz=[]
    # Parcourir tous les fichiers CSV dans le dossier
    plt.figure()
    
    for fichier in fichiers_csv:
        
        chemin_fichier = os.path.join(dossier, fichier)

        # Calculer les moyennes pour le fichier actuel
        moy_fichier, freq_fichier = moyenne_impedance_fichier(chemin_fichier)
        #moy_phase, freq_phase = moyenne_phase_fichier(chemin_fichier)
        # Utilisation de différentes couleurs avec des lignes pointillées si nécessaire
        magn_20khz = moy_fichier[62]
        moyenne_20khz.append(magn_20khz)
        #style_ligne = '-' if len(moy_fichier) > 10 else '-'
        plt.plot(freq_fichier, moy_fichier, linestyle='-', label=fichier)
        moyennes_dossier.append(moy_fichier)
        moyennes_dossier[0].append(fichier)


    #print("valeurs pour csv " , moyennes_dossier)
    plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(0, 1000)  # Définir les limites de l'axe y
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Moyenne de l\'impédance (ohms)')
    #plt.title('Moyennes de l\'impédance de tous les fichiers en fonction de la fréquence (échelle log)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=len(fichiers_csv))
    plt.show()
    
    
    return moyennes_dossier, freq_fichier

#################################################################################################   
#                                           IMP VS CONCENTRATION                                #
################################################################################################# 

def csv_for_scilab(nom, moyennes_dossier, frequences): ##### pour avoir les moyennes de chaque mesure d'un dossier pour toutes les frequences, avec comme premiere clonne le nom de la mesure
        nom_fichier = nom + '.csv'

        try:
            with open(nom_fichier, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=';')

                # Write frequencies in the first row
                #csvwriter.writerow(frequences + [frequences[0]])
                csvwriter.writerow( ['']+frequences )

                # Write data starting from the third row
                for i, data_row in enumerate(moyennes_dossier[1:], start=1):
                    # Check if the index is within the range of moyennes_dossier[0]
                    label = moyennes_dossier[0][i-1] if i-1 < len(moyennes_dossier[0]) else ''
                    #csvwriter.writerow(data_row + [label])
                    csvwriter.writerow([label] + data_row)

            print(f"Les données ont été ajoutées avec succès à {nom_fichier}.")
        except Exception as e:
            print(f"Une erreur s'est produite lors de l'écriture des données CSV : {e}")
            
def imp_points_freq_pour_concentr(dossier,param):
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

    moyennes, frequences = moyennes_dossier(dossier)
    print('moyennes :', moyennes)

    freq_indices = [19, 54, 90]
    freq = [frequences[i] for i in freq_indices]

    print('\n----- Impedance vs concentration sur 3 fréq (1k, 10k, 100k) ------')
    print('etude sur ces fréquences :', freq)
    print('moyennes : ' ,moyennes)
    points = [[] for _ in range(len(freq))]
    
    for j in range(len(points)):
        if len(param)==2:
            points[j].append(moyennes[1][freq_indices[j]])
            points[j].append(moyennes[2][freq_indices[j]])
        elif len(param)==3:
            points[j].append(moyennes[1][freq_indices[j]])
            points[j].append(moyennes[2][freq_indices[j]])
            points[j].append(moyennes[3][freq_indices[j]])
        tri_croissant(points[j])
    print('Valeurs impédance des conditions:', points)
    return points

def imp_vs_concentration(dossier,param,color):
    #plt.figure()
    
    if param=='alcool':
        c=[0, 71 , 141]
        
    elif param=='azote':
        c=[500, 250, 0]  
        
    elif param=='sucre':
        c=[0, 150, 300]
    
    elif param=='fermentation':
        c=[0,2,10]
    
    elif param=='sucre2':
        c=[0,150]
        
    elif param=='alcool2':
        c=[0,71]
        
    elif param=='azote2':
        c=[250,0]
    
    points= imp_points_freq_pour_concentr(dossier,c)

    plt.plot(c, points[0], linestyle='-',color=color)
    plt.plot(c, points[1], linestyle='--',color=color)
    plt.plot(c, points[2], linestyle='-.',color=color)
        
    plt.title('imp_vs_concentration')
    plt.xlabel(f"concentration (mg/L) de {param}")
    plt.ylabel('impedance en ohms')
    plt.yscale('log')
    plt.ylim(100, 120000)  # Définir les limites de l'axe y
    plt.legend()
    plt.grid(True)
    plt.show()
  
def read_first_column(file_name):
    result = []

    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        
        for row in reader:
            # Check if the first cell in the row is not empty
            if row and row[0]:
                result.append(row[0])
            else:
                break  # Stop reading if an empty cell is encountered

    return result

#################################################################################################   
#                                           IMP VS TEMPS                                        #
################################################################################################# 

def creer_fichier_csv(imp, freq, nom_fichier):
    # Vérifier que les listes imp et frequences_etudiees ont la même longueur
    nom_fichier=nom_fichier+".csv"
    # Vérifier que les tableaux ont la même longueur
    if len(freq) != len(imp):
        raise ValueError("Les tableaux freq et imp doivent avoir la même longueur.")
    print("nb mesures",len(imp[0]))
    # Écrire les données dans le fichier CSV
    with open(nom_fichier, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')

        # Écrire la première ligne avec les en-têtes
        entetes = ['Frequences'] + [f"mesure n°{j}" for j in range(len(imp[0])) for i in range(len(imp))]
        csvwriter.writerow(entetes)

        # Écrire les lignes de données
        for i in range(len(freq)):
                ligne = [freq[i]] + [imp[i][j] for j in range(len(imp[i]))]
                csvwriter.writerow(ligne)
    print("Ecriture finie")

def indice_valeur_plus_proche(liste, valeur):
    return min(range(len(liste)), key=lambda i: abs(liste[i] - valeur)) 

def suivi_duree(fichier, freq_a_etudier):
    # calculer la moyenne des magnitudes de toutes les mesures d'un fichier à chaque fréquence
    # Récupérer les données du fichier  
    donnees_fichier = recuperer_donnees_fichier(fichier)
    
    # Vérifier si des données ont été récupérées
    if donnees_fichier is None:
        return None

    # stockage impedance
    imp_magn = []
    imp_phase = []
    # Lignes a etudier
    lignes_magnitude = []
    lignes_phase = []
    
    # Vérifier si freq_a_etudier est une liste
    if not isinstance(freq_a_etudier, list):
        freq_a_etudier = [freq_a_etudier]
        
    indices_freq = []
    frequences = string_to_float(donnees_fichier[1])
    
    for freq in freq_a_etudier:
        index = indice_valeur_plus_proche(frequences, freq)
        indices_freq.append(index)
        
        # tableaux 2 dimensions
    for _ in freq_a_etudier:
        imp_magn.append([])
    for _ in freq_a_etudier:
        imp_phase.append([])
    
    nb_lignes = len(donnees_fichier)
    nb_mesures = int(nb_lignes / 4)
    nb_colonnes = len(donnees_fichier[1])

    for i in range(nb_mesures):
        lignes_magnitude.append(2 + 4 * i)
        lignes_phase.append(3 + 4*i)

    # Ajout de listes vides pour chaque fréquence dans imp_magn
    # for _ in frequences_etudiees:
    #     imp_magn.append([])
    
    print('len(imp_magn) :',len(imp_magn))
    print('nb frequences :',len(indices_freq))
    print('freq a etudier : ', freq_a_etudier)
    print('lignes phase :', lignes_phase)
    print('lignes magnitude:', lignes_magnitude)
    a=0
    for colonne in indices_freq:
        
        for ligne in lignes_magnitude:
            imp_magn[a].append(round(float(donnees_fichier[ligne][colonne]), 2))
            imp_phase[a].append(round(float(donnees_fichier[ligne+1][colonne]), 2))
        # for line in lignes_phase:
        #     imp_phase[a].append(round(float(donnees_fichier[ligne][colonne]), 2))
        a+=1
    # Plot des points
    # plt.figure()
    # for i, freq in enumerate(frequences_etudiees):
    #     plt.plot(imp_magn[i], label=f'Fréquence {freq} Hz')

    # Configuration des axes et affichage des légendes
    # plt.xscale('log')
    # plt.xlabel('Mesure')
    # plt.ylabel('Impédance magnitude (moy)')
    # plt.grid(True)
    # plt.ylim(0, 1000)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
    #plt.show()
    
    return imp_magn,imp_phase, frequences, freq_a_etudier

def afficher_derivee(imp_magn, frequences, fenetre_moyenne):
    # Durée entre chaque point en secondes
    duree_entre_points = 590  # 9 minutes et 50 secondes

    # Initialisation du temps
    temps_actuel = timedelta(seconds=0)
    
    # Création du graphique
    plt.figure(figsize=(10, 6))

    # Création de la liste de temps une seule fois
    temps = []
    for j in range(len(imp_magn[0])):
        temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
        temps_actuel += timedelta(seconds=duree_entre_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calcul de la moyenne glissante
        moyenne_glissante = np.convolve(magnitudes, np.ones(fenetre_moyenne)/fenetre_moyenne, mode='valid')

        # Calcul de la dérivée numérique
        derivee = np.gradient(moyenne_glissante, temps[:len(moyenne_glissante)])

        # Affichage de la dérivée avec une couleur différente pour chaque fréquence
        plt.plot(temps[:len(moyenne_glissante)], derivee, marker='o', linestyle='-', label=f'Dérivée - Fréquence {frequences[i]}')

    plt.title('Dérivée des Points en fonction du temps')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Dérivée des Magnitudes')
    plt.legend()
    plt.grid(True)
    
def impedance_vs_heures(imp_magn,frequences,frequences_etudiees,fenetre_moyenne):
    # Durée entre chaque point en secondes
    duree_entre_points = 590  # 9 minutes et 50 secondes

    # Initialisation du temps
    temps_actuel = timedelta(seconds=0)
    
    # Liste de couleurs pour chaque fréquence
    couleurs = ['b', 'g', 'r', 'c', 'm','y','grey', 'black', 'pink']
    impedance_to_return=[]
    # Création du graphique
    plt.figure(figsize=(10, 6))
    # Création de la liste de temps une seule fois
    temps = []
    for j in range(len(imp_magn[0])):
        temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
        temps_actuel += timedelta(seconds=duree_entre_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calcul de la moyenne glissante
        
        
        ############################ type de filtrage a appliquer ###########################################
        
        ### ------------------------------------------------------------------- pas de filtrage 
        filtered_magnitudes=magnitudes
        
        
        
        ### ------------------------------------------------------------------- moyenne glissante 
        moyenne_glissante = np.convolve(filtered_magnitudes, np.ones(fenetre_moyenne)/fenetre_moyenne, mode='valid')
        print("moy glissante : ",moyenne_glissante)
        ### ------------------------------------------------------------------- filtrer les points bas
        max_threshold = 30
        window_size = 3
        #filtered_magnitudes = filter_low_values(magnitudes, max_threshold, window_size)
        
        ###### pour filtrer les points bas (4 prochaines lignes)
        # max_threshold = 20
        # window_size = 10
        # filtered_magnitudes = filter_low_values(moyenne_glissante, max_threshold, window_size)
        # plt.plot(temps[:len(filtered_magnitudes)], filtered_magnitudes, marker='o',markersize=1, linestyle='-', label=str(freq_khz) + ' kHz')

        # Affichage de la ligne avec une couleur différente pour chaque fréquence
        freq=round(frequences_etudiees[i])
        freq_khz = round(freq/1000,2)
        plt.plot(temps[:len(moyenne_glissante)], moyenne_glissante, marker='o',markersize=2, linestyle='-', color=couleurs[i], label=str(freq_khz) + ' kHz')
        impedance_to_return.append(moyenne_glissante)
    plt.title(f'675 points : Impedance vs temps (moyenne glissante fenetre {fenetre_moyenne} pts)')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Moyenne Glissante impedance (ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return impedance_to_return

def impedance_vs_secondes(imp_magn, frequences, frequences_etudiees):
    # Durée entre chaque point en secondes
    duree_entre_points = 8

    # Initialisation du temps
    temps_actuel = timedelta(seconds=0)

    # Liste de couleurs pour chaque fréquence
    couleurs = ['b', 'g', 'r', 'c', 'm']

    # Création du graphique
    plt.figure(figsize=(10, 6))

    # Création de la liste de temps une seule fois
    temps = []
    for j in range(len(imp_magn[0])):
        temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
        temps_actuel += timedelta(seconds=duree_entre_points)

    for i, magnitudes in enumerate(imp_magn):
        # Affichage de la ligne avec une couleur différente pour chaque fréquence
        plt.plot(temps[:len(magnitudes)], magnitudes, marker='o', linestyle='-', color=couleurs[i], label=frequences[frequences_etudiees[i]])

    plt.title('Impedance vs Temps (Toutes les 8 secondes)')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Impedance (ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()

def derivee_vs_heures(imp_magn, frequences,frequences_etudiees):
    # Durée entre chaque point en secondes
    duree_entre_points = 590  # 9 minutes et 50 secondes

    # Initialisation du temps
    temps_actuel = timedelta(seconds=0)
    
    # Liste de couleurs pour chaque fréquence
    couleurs = ['b', 'g', 'r', 'c', 'm']

    # Création du graphique
    plt.figure(figsize=(10, 6))
    fenetre_moyenne=1
    # Création de la liste de temps une seule fois
    temps = []
    for j in range(len(imp_magn[0])):
        temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
        temps_actuel += timedelta(seconds=duree_entre_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calcul de la moyenne glissante
        moyenne_glissante = np.convolve(magnitudes, np.ones(fenetre_moyenne)/fenetre_moyenne, mode='valid')
        freq=round(frequences[frequences_etudiees[i]])
        # Affichage de la ligne avec une couleur différente pour chaque fréquence
        
        derivee = np.gradient(moyenne_glissante, temps[:len(moyenne_glissante)])

        # Affichage de la dérivée avec une couleur différente pour chaque fréquence
        plt.plot(temps[:len(moyenne_glissante)], derivee, marker='o', linestyle='-', color=couleurs[i], label=freq)
        #plt.plot(temps[:len(moyenne_glissante)], moyenne_glissante, marker='o', linestyle='-', color=couleurs[i], label=freq)

    plt.title('derivee des Points en fonction du temps')
    plt.xlabel('Temps (heures)')
    plt.ylabel('derivees des Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.show()

def dossier_vs_secondes(dossier):
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

    if not fichiers_csv:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None

    # Initialisation des listes pour stocker les données
    imp_magn_dossier = []
    frequences_dossier = []
    frequences_etudiees_dossier = []

    # Parcourir tous les fichiers CSV dans le dossier
    for fichier in fichiers_csv:
        # Vérifier si le nom de fichier se termine déjà par '.csv'
        if not fichier.endswith('.csv'):
            fichier = fichier + '.csv'
            
        chemin_fichier = os.path.join(dossier, fichier)

        # Calculer les données pour le fichier actuel
        imp, freq, freq_etudiees = suivi_duree(chemin_fichier)

        # Ajouter les données du fichier actuel aux listes
        imp_magn_dossier.append(imp)
        frequences_dossier.append(freq)
        frequences_etudiees_dossier.append(freq_etudiees)

    # Créer le graphique pour les mesures du dossier
    plt.figure(figsize=(12, 8))

    # Appeler la fonction impedance_vs_secondes pour chaque fichier
    for i in range(len(fichiers_csv)):
        impedance_vs_secondes(imp_magn_dossier[i], frequences_dossier[i], frequences_etudiees_dossier[i])

    # Ajouter des étiquettes et légendes
    plt.title('Impedance vs Temps pour Chaque Fichier (Toutes les 8 secondes)')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Impedance (ohms)')
    plt.legend(title='Fichiers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

def variation_vs_heures(imp_magn, frequences,frequences_etudiees):
        # Durée entre chaque point en secondes
        duree_entre_points = 590  # 9 minutes et 50 secondes

        # Initialisation du temps
        temps_actuel = timedelta(seconds=0)
        
        # Liste de couleurs pour chaque fréquence
        couleurs = ['b', 'g', 'r', 'c', 'm']

        # Création du graphique
        plt.figure(figsize=(10, 6))
        fenetre_moyenne=40
        # Création de la liste de temps une seule fois
        temps = []
        for j in range(len(imp_magn[0])):
            temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
            temps_actuel += timedelta(seconds=duree_entre_points)
        
        fenetre_moyenne = 1
        for i, magnitudes in enumerate(imp_magn):
            moyenne_glissante = np.convolve(magnitudes, np.ones(fenetre_moyenne)/fenetre_moyenne, mode='valid')
            # Calcul de la moyenne glissante
            premier_point = moyenne_glissante[0]
            pourcentage_variation = [(point - premier_point) / premier_point * 100 for point in moyenne_glissante]

            # Affichage de la courbe de pourcentage de variation avec une couleur différente pour chaque fréquence
            plt.plot(temps[:len(moyenne_glissante)], pourcentage_variation, marker='o', linestyle='-', label=f'Variation - Fréquence {frequences[i]}')

        plt.title('Moyenne Glissante des Points en fonction du temps')
        plt.xlabel('Temps (heures)')
        plt.ylabel('Moyenne Glissante des Magnitudes')
        plt.legend()
        plt.grid(True)
        plt.show()

def filter_low_values(magnitudes, max_threshold, window_size):
    """
    Filtre les valeurs en éliminant celles en dessous de la valeur maximale moins 50 ohms dans des paquets de 100 ou 1000 valeurs.

    Parameters:
    - magnitudes (list): Liste des magnitudes à filtrer.
    - max_threshold (int): Seuil maximal pour le filtrage.
    - window_size (int): Taille de la fenêtre pour le filtrage.

    Returns:
    - filtered_magnitudes (list): Liste des magnitudes filtrées.
    """

    filtered_magnitudes = []

    for i in range(0, len(magnitudes), window_size):
        window = magnitudes[i:i+window_size]

        # Appliquer le filtre
        max_value = np.max(window)
        filtered_window = [val if val >= (max_value - 50) else np.nan for val in window]

        # Ajouter les valeurs filtrées à la liste résultante
        filtered_magnitudes.extend(filtered_window)

    return filtered_magnitudes

########################## fonctions pour le filtrage passant par complexes

def impedance_complex(module, phase_deg): ## transfo magnitude et phase en impédance complexe
    phase_rad = math.radians(phase_deg)
    impedance_complexe =cmath.rect(module, phase_rad)
    
    return impedance_complexe

def retrouver_phase_magnitude(impedance): ## recupérer valeurs de magnitude et de phase 
    phase_radians = cmath.phase(impedance)
    phase_degrees = math.degrees(phase_radians)
    # if phase_degrees>100:
    #     phase_degrees=phase_degrees-180
    magnitude = abs(impedance)
    
    return magnitude, phase_degrees

def tab_to_complex(modules, phases_deg):
    complex_impedance = []

    for i in range(len(modules)):
        row = []
        for j in range(len(modules[i])):
            impedance = impedance_complex(modules[i][j], phases_deg[i][j])
            row.append(impedance)
        complex_impedance.append(row)

    return complex_impedance

def moyenne_complexes(complexes):
    if not complexes:
        raise ValueError("La liste de nombres complexes ne doit pas être vide.")

    somme = sum(complexes)
    moyenne = somme / len(complexes)

    return moyenne

def moyenne_complexe_fichier(fichier): # calculer la moyenne des magnitudes de toutes les mesures d'un fichier à chaque fréquence
    # Récupérer les données du fichier  
    donnees_fichier = recuperer_donnees_fichier(fichier)
    #print( "données fichier: ",donnees_fichier)
    # Vérifier si des données ont été récupérées
    if donnees_fichier is None:
        return None

    # Liste pour stocker les moyennes
    complexe=[]
    
    moyennes_magnitude = []
    moyennes_phase = []
    moyennes = []

    # Lignes à moyenner
    lignes_magn = []
    lignes_phase =[]
    
    frequences=string_to_float(donnees_fichier[1])
    
    nb_lignes=len(donnees_fichier)
    nb_mesures=int(nb_lignes/4)
    nb_colonnes=len(donnees_fichier[1])
    #print("fichier : ", fichier)
    #print("nb mesures", nb_mesures)
    for i in range(nb_mesures):
        lignes_magn.append(2+4*i)
        lignes_phase.append(3+4*i)
    #print('lignes a moyenner',lignes_a_moyenner)
    #print('nb lignes fichier :',nb_lignes)
    #print('frequences:', frequences)
    #print('nb pt freq (colonnes) : ', len(frequences))
    
    for colonne in range(len(frequences)):## pour chaque frequence
        sum_imp=0
        for ligne in lignes_magn:
            magn=(round(float(donnees_fichier[ligne][colonne]),2))
            phase=(round(float(donnees_fichier[ligne+1][colonne]),2))
            complexe.append(impedance_complex(magn, phase))      # tableau contenant tous les complexes pour 1 frequence
        moy_comp=moyenne_complexes(complexe)                     # calcul de la moyenne du tableau
        moy_magn,moy_phase=retrouver_phase_magnitude(moy_comp)   # moyenne en non complexe
        
        moyennes_magnitude.append(moy_magn)                     # ajout de la moyenne pour chaque freq
        moyennes_phase.append(moy_phase)             
            # sum_imp = sum_imp + round(float(donnees_fichier[ligne][colonne]),2)
            # moy_col=round((sum_imp*4)/nb_lignes,1)
        #moyennes.append(moy_col)
        
    # print("Moyennes des magn a chaque freq:",moyennes_magnitude)
    # print("Moyennes des phases a chaque freq:",moyennes_phase)

    #print('nb de moy :', len(moyennes))
    return moyennes_magnitude,moyennes_phase, frequences   

def moyennes_par_complexe_dossier(dossier):
    # Obtenir la liste des fichiers CSV dans le dossier
    fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]

    # Vérifier s'il y a au moins un fichier CSV dans le dossier
    if not fichiers_csv:
        print("Aucun fichier CSV trouvé dans le dossier.")
        return None

    # Créer une figure avec deux sous-graphiques (pour l'impédance et la phase)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Parcourir tous les fichiers CSV dans le dossier
    for fichier in fichiers_csv:
        chemin_fichier = os.path.join(dossier, fichier)

        # Calculer les moyennes pour le fichier actuel (impédance et phase)
        # moy_impedance, freq_impedance = moyenne_impedance_fichier(chemin_fichier)
        # moy_phase, freq_phase = moyenne_phase_fichier(chemin_fichier)
        moy_impedance,moy_phase,frequence=moyenne_complexe_fichier(chemin_fichier)

        # Tracer l'impédance sur le premier sous-graphique
        axes[0].plot(frequence, moy_impedance, linestyle='-', label=fichier)

        # Tracer la phase sur le deuxième sous-graphique
        axes[1].plot(frequence, moy_phase, linestyle='-', label=fichier)

    # Configuration des axes et affichage des légendes
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Fréquence (Hz)')
    axes[0].set_ylabel('Impédance magnitude (Ω)')
    axes[0].grid(True)
    #axes[0].set_ylim(0, 2000)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('Fréquence (Hz)')
    axes[1].set_ylabel('Impédance phase (degrés)')
    axes[1].grid(True)

    # Ajuster la disposition pour un meilleur espacement
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
    
def impedance_vs_heures_par_complexe(imp_magn,imp_phase,frequences_etudiees,fenetre_moyenne,etude_complexe):
    print("--------------------------------------------------------------------")
    print("------------- starting following impedance vs hours ----------------")
    print("--------------------------------------------------------------------")
    # Durée entre chaque point en secondes
    duree_entre_points = 590  # 9 minutes et 50 secondes

    # Initialisation du temps
    temps_actuel = timedelta(seconds=0)
    
    # Liste de couleurs pour chaque fréquence
    couleurs = ['b', 'g', 'r', 'c', 'm','y','grey', 'black', 'pink']
    plt.figure(figsize=(10, 6))
    # Création de la liste de temps une seule fois
    temps = []
    
    impedance_to_return=[]
    for j in range(len(imp_magn[0])):
        temps.append(temps_actuel.total_seconds() / 3600)  # Convertir le temps en heures
        temps_actuel += timedelta(seconds=duree_entre_points)
    
    if etude_complexe=='yes':
        tab_complex = tab_to_complex(imp_magn, imp_phase)
        print("Impédances converties en complexe avec succès")
    else:
        tab_complex = imp_magn
        
    
    print(f"Analyse de {len(tab_complex[0])} points à {frequences_etudiees} Hz")
    for i, imp_complex in enumerate(tab_complex):
        
        moyenne_glissante = []
        
        ############################ type de filtrage a appliquer ###########################################
        
    
        ### MOYENNE GLISSANTE  ------------------------------------------------------------------- 
        
        moyenne_glissante_complex = np.convolve(imp_complex, np.ones(fenetre_moyenne)/fenetre_moyenne, mode='same')
        
        if fenetre_moyenne==1:                                                                                      ####
            print("Pas de moyennage glissant")                                                                      ####
        else:                                                                                                       ####
            print(f"Moyenne glissante à {frequences_etudiees[i]} Hz avec fenetre de {fenetre_moyenne} pts : ok")    ####
            
            
        # print("moy gliss complex : ",moyenne_glissante_complex)               # pour verifier les impédances obtenues
        # print("len(moy gliss complex) :", len(moyenne_glissante_complex))
        
        ### retourner en magnitude : 
        for j in range(len(moyenne_glissante_complex)):
            magn_moy,phase=retrouver_phase_magnitude(moyenne_glissante_complex[j])
            moyenne_glissante.append(round(magn_moy,2))
        # print("Moyenne glissante : ", moyenne_glissante)
        # print("len(moy gliss)    : ", len(moyenne_glissante))
        
        ### filtrer les points bas ------------------------------------------------------------------- 
        max_threshold = 30
        window_size = 3
        #magnitudes = filter_low_values(moyenne_glissante, max_threshold, window_size)
        
        ###### pour filtrer les points bas (4 prochaines lignes)
        # max_threshold = 20
        # window_size = 10
        # filtered_magnitudes = filter_low_values(moyenne_glissante, max_threshold, window_size)
        # plt.plot(temps[:len(filtered_magnitudes)], filtered_magnitudes, marker='o',markersize=1, linestyle='-', label=str(freq_khz) + ' kHz')

        # Affichage de la ligne avec une couleur différente pour chaque fréquence
        freq=round(frequences_etudiees[i])
        freq_khz = round(freq/1000,2)
        plt.plot(temps[:len(moyenne_glissante)], moyenne_glissante, marker='o',markersize=2, linestyle='-', color=couleurs[i], label=str(freq_khz) + ' kHz')
        impedance_to_return.append(moyenne_glissante)
    plt.title(f'675 points : Impedance vs temps (moyenne glissante PAR COMPLEXE fenetre {fenetre_moyenne} pts)')
    plt.xlabel('Temps (heures)')
    plt.ylabel('Moyenne Glissante impedance (ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return impedance_to_return, temps
    
def plot_difference(array1, array2):
    plt.figure(figsize=(10, 6))
    # Vérifier que les deux tableaux ont la même longueur
    print('taille des tableaux :', len(array1))
    if len(array1) != len(array2):
        raise ValueError("Les deux tableaux doivent avoir la même longueur.")

    # Calculer la différence entre les valeurs correspondantes
    differences = np.subtract(array1, array2)

    # Créer un tableau d'indices pour le nombre de valeurs
    indices = np.arange(len(array1))

    # Tracer le graphe des différences en fonction du nombre de valeurs
    plt.plot(indices, differences, marker='o', linestyle='-', color='b')
    plt.xlabel('Nombre de valeurs')
    plt.ylabel('Différence')
    plt.title('Différence entre les valeurs des deux tableaux')

    # Afficher le graphe sur une fenêtre séparée
    plt.show()

#################################################################################################   
#                                           recup debitmetre                                    #
################################################################################################# 
def recup_vitesse_CO2(file):
    # Obtenez le chemin absolu du répertoire du script
    repertoire_script = os.path.dirname(os.path.abspath(__file__))
    
    # Concaténez le nom du fichier texte au chemin du répertoire du script
    chemin_fichier = os.path.join(repertoire_script, file)
    
    # Initialisez une liste pour stocker chaque élément de la quatrième colonne de chaque ligne du fichier
    donnees_fichier = []
    
    # Ouvrez le fichier en mode lecture
    with open(chemin_fichier, 'r') as fichier:
        # Parcourez chaque ligne du fichier
        for ligne in fichier:
            # Divisez la ligne en mots (ou éléments séparés par un espace)
            mots = ligne.split()
            # Si la ligne a au moins 4 mots, ajoutez le 4ème mot à la liste des données
            if len(mots) >= 4:
                donnees_fichier.append(mots[3].strip())
    
    # Supprimez les chaînes 'Vitesse' et 'phase'
    donnees_fichier.remove('Vitesse')
    donnees_fichier.remove('phase')
    
    # Convertissez les éléments restants en nombres à virgule flottante
    donnees_fichier = [float(valeur) for valeur in donnees_fichier]
    
    return donnees_fichier

def plot_vitesse_CO2(file):
    # Récupérer les données
    donnees = recup_vitesse_CO2(file)

    # Générer les temps en fonction du nombre de données (20 minutes entre chaque point)
    temps = [i * 20 / 60 for i in range(len(donnees))]

    # Créer une figure avec un axe partagé à droite
    fig, ax1 = plt.subplots()

    # Tracer les données sur l'axe des y à gauche
    ax1.plot(temps, donnees, label='Vitesse CO2', color='blue')
    ax1.set_xlabel('Temps (heures)')
    ax1.set_ylabel('Vitesse CO2', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Créer un axe partagé à droite avec une échelle différente
    ax2 = ax1.twinx()
    ax2.set_ylabel('Y Axe à Droite', color='red')
    ax2.plot(temps, [2 * valeur for valeur in donnees], label='Exemple Y Axe à Droite', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Vitesse CO2 en fonction du temps')
    plt.show()

def plot_impedance_and_velocity(file, imp_magn, imp_phase, frequences_etudiees, fenetre_moyenne, etude_complexe):
    # Récupérer les données de vitesse CO2
    donnees_vitesse_CO2 = recup_vitesse_CO2(file)

    # Récupérer les données d'impédance
    donnees_impedance, temps = impedance_vs_heures_par_complexe(imp_magn, imp_phase, frequences_etudiees, fenetre_moyenne, etude_complexe)

    # Générer les temps en fonction du nombre de données (20 minutes entre chaque point)
    temps_vitesse_CO2 = [i * 20 / 60 for i in range(len(donnees_vitesse_CO2))]

    # Créer une figure
    fig, ax1 = plt.subplots()

    # Tracer les données sur l'axe des y à gauche
    ax1.plot(temps_vitesse_CO2, donnees_vitesse_CO2, label='Vitesse CO2', color='blue')
    ax1.set_xlabel('Temps (heures)')
    ax1.set_ylabel('Vitesse CO2', color='blue')
    ax1.set_xlim(2, 108)
    
    ax1.tick_params(axis='y', labelcolor='blue')

    # Créer un axe partagé à droite avec une échelle différente
    ax2 = ax1.twinx()
    ax2.set_ylabel('Y Axe à Droite', color='red')
    ax2.set_xlim(2, 108)
    ax2.set_ylim(250, 500)
    # Tracer les données d'impédance en utilisant la première ligne
    ax2.plot(temps, donnees_impedance[0], label='Impedance', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Ajouter une légende combinée pour les deux axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Vitesse CO2 en fonction du temps')
    plt.show()



################################################################################################      
###--------------------------------------MAIN-----------------------------------------------####
################################################################################################   

#def main():
     
    #remplir_csv('testtttt.csv', 3, 10)
    #dossier = input('Dossier contenant les mesures: ')
    
    # dossier='test_agitation_carte3_moyennes'
    # moyennes_dossier(dossier)                   #moyenner les valeurs des fichiers du dossier
    # moyennes_par_complexe_dossier(dossier)      #moyenner en passant par complexes
    
    
    ###########################################     afficher les courbes d'impédance vs fréquence des mesures du dossier
    #print_moyennes_dossier(dossier)
    
    
   ############################################     creer un fichier contenant les moyennes
   # filename = input('nom csv pour scilab : ')          
   # csv_for_scilab(filename, moyennes, frequences)      
   
   ############################################     afficher impedance vs concentration (sur des pts en frequences)
   #imp_points_freq_pour_concentr(dossier,param)
   
   ## afficher l'impédance en fonction de la concentration
   # parametre = input('paramètre étudié :')
   # color = input('color:')
   #imp_vs_concentration(dossier,parametre,color)
   
   
   # #########################################      suivi de fermentation  
    
    # file='FERMENTATION.csv'
    # freq_a_etudier=[20000]
    # fenetre_moyennage=40
    # etude_complexe='yes'    # yes pour activer le passage en complexe avant le filtrage
    
    # imp,phase, frequences,frequences_etudiees = suivi_duree(file,freq_a_etudier)
    
    # print('freq_a_etudier: ',freq_a_etudier)
    # print("impedances : ",imp)
    # print(f'phase :', phase)
    
    # tab_complex = tab_to_complex(imp, phase)
    # print("Tableau des impedances complexes",tab_complex)
    
    # moy1=impedance_vs_heures(imp, frequences,frequences_etudiees,fenetre_moyennage)
    # moy2=impedance_vs_heures_par_complexe(imp,phase, freq_a_etudier,fenetre_moyennage,etude_complexe)
    # print("Moy1 : ",len(moy1))
    # print("Moy2 : ",len(moy2))

    #debitmetre = recup_vitesse_CO2('R1.txt')
    #print(debitmetre)
    # Utilisation de la fonction pour récupérer et tracer les données
    
    # plot_vitesse_CO2('R1.txt')
    # plot_impedance_and_velocity('R1.txt', imp, phase, frequences_etudiees, fenetre_moyennage, etude_complexe)
    
    #plot_difference(moy1, moy2)
    
    ################### rentrer les imp dans fichier csv pour analyse depuis excel
    #file_excel=input('Dans quel fichier envoyer les impédances ? : ')
    #creer_fichier_csv(imp,frequences_etudiees,file_excel)
    #dossier_vs_secondes(dossier)
    
   # #########################################      analyse temperature
   # file_celcius=input("Fichier temperature")
   # imp2, frequences2,frequences_etudiees2 = suivi_duree(file_celcius)
   # print("impedances temperature : ",imp2)
   # afficher_points_temps(imp2, frequences2,frequences_etudiees2)
   # #print("frequences : ",frequences2)
   
   ########### selection frequence
    #imp2, freq2 = impedance_a_1_freq(file, 40000)
    #print(f'imp2 at {freq2}:', imp2)

################################################################################################      
###---------------------------------MAINS UTILES--------------------------------------------####
################################################################################################   


# def main(): # main pour afficher l'impedance + vitesse de dégagement CO2 en fonction du temps
#     file='FERMENTATION.csv'
#     freq_a_etudier=[20000]
#     fenetre_moyennage=40
#     etude_complexe='yes'    # yes pour activer le passage en complexe avant le filtrage
#     imp,phase, frequences,frequences_etudiees = suivi_duree(file,freq_a_etudier)
    
#     plot_vitesse_CO2('R1.txt')
#     plot_impedance_and_velocity('R1.txt', imp, phase, frequences_etudiees, fenetre_moyennage, etude_complexe)
    
#     imp,phase, frequences,frequences_etudiees = suivi_duree(file,freq_a_etudier)
 

# ----------------------------------------------------------------------------------------------
def main(): # plot les courbes d'impédance (magnitude et phase) vs frequence de tous les fichiers dans un dossier
        
        #dossier = input('Dossier contenant les mesures: ')      
        #dossier='test_agitation_carte3_moyennes'
        dossier='sucre_18janvier'
        moyennes_par_complexe_dossier(dossier)      #moyenner en passant par complexes
        #print_moyennes_dossier(dossier)
 
# ----------------------------------------------------------------------------------------------
   
if __name__ == "__main__":  
    main()
    