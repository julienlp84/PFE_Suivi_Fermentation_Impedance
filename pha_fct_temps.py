# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:56:46 2022

@author: valentin.lefebvre01
"""


from plot_print import tableau_tps_reel
from correction import avancement
#import math as m


"""
lit un fichier avec plusieur balayage en frequence repartie dans le temps 
en un fichier avec la magnitude en fct du temps selon differente frequence avec moyenne glissante

@param nom_fichier: le nom du fichier.csv avec plusieur balayage en frequence repartie dans le temps
"""
def  pha_fct_freq(nom_fichier):
    f = open(nom_fichier,'r')
    data = f.read()
    data_split = data.split('\n')
    tram_freq = data_split[1].split(';')
    tram_freq = tram_freq[:-1]
    nb_data = len(tram_freq)
    f.close()
    pha_fct_temps = [0]*nb_data
    pha_fct_temps_corr = [0]*nb_data
    #print(tram_freq)
    intervalle_moy_glissante = 21 #nombre de point pour la moyenne glissante
    assert intervalle_moy_glissante%2 == 1 , 'intervalle_moy_glissante doit etre un nombre impaire'
    pha_fct_temps_moyen = [0]*nb_data
    print('building data at frequency: ')
    
    ecart_mesure_max = 5
    for i in range(nb_data): 
        """ analyse du fichier et recuperation de la magnitude selon chaque frequence balayé"""
        [tps, Mag_graph, Pha_graph] = tableau_tps_reel(nom_fichier, float(tram_freq[i]))
        pha_fct_temps[i] = Pha_graph
        pha_fct_temps_corr[i] = erreur_mesure(Pha_graph, ecart_mesure_max)
        pha_fct_temps_corr[i] = erreur_mesure(pha_fct_temps_corr[i], ecart_mesure_max)
        pha_fct_temps_moyen[i] = moyenne_glissante(pha_fct_temps_corr[i], intervalle_moy_glissante)
    
    print('creating file tableau_pha_tps_reel.csv')
    f = open('tableau_pha_tps_reel.csv','a')
    
    f.write('temps_en_sec')
    
    """ ecriture des données sans moyenne glissante"""
    for i in range(len(tps)):
        f.write(';' + str(tps[i]))
    f.write('\n')
    avancee = 0
    print('writing data without average')
    for j in range(nb_data):
        f.write(str(tram_freq[j]) + 'Hz')
        for k in range(len(tps)):
            f.write(';' + str(pha_fct_temps[j][k]))
        f.write('\n')
        avancee = avancement(j, nb_data, avancee)
    
    f.write('\n\n')
    
    print('writing data with average')
    """ ecriture des données avec moyenne glissante"""
    for j in range(nb_data):
        f.write(str(tram_freq[j]) + 'Hz')
        f.write(';'*(intervalle_moy_glissante//2))
        for k in range(len(pha_fct_temps_moyen[j])):
            f.write(';' + str(pha_fct_temps_moyen[j][k]))
        f.write('\n')
        avancee = avancement(j, nb_data, avancee)
        
        
    f.close()
    print('tableau_pha_tps_reel.csv has successfully been built')
    

"""
enleve les erreurs de mesures (pic lors d'une mesure)
                               
@param value: un tableau de valeur en fct du temps 
@param saut_max: le saut maximum autorisé entre deux mesures |mesure precedente - mesure actuelle| + |mesure suivante - mesure actuelle|

@return value_corr: un tableau des valeurs sans pics de mesures indesirés
"""    
def erreur_mesure(value,saut_max):
    
    pic = 0.0
    value_corr = [0]*len(value)
    value_corr[0] = float(value[0])
    for i in range(len(value) - 2):
        pic = abs(float(value[i])-float(value[i+1])) + abs(float(value[i+1])-float(value[i+2]))
        if pic > saut_max:
            value_corr[i+1] = (value[i] + value[i+2])/2
        else:
            value_corr[i+1] = value[i+1]
        
    value_corr[-1] = value[-1]
    
    return value_corr

"""
moyenne glissante d'un jeu de valeurs selon un intervalle donné

@param valeurs: le jeu de valeur sur lequelle on souhaite faire une moyenne glissante
@param intervalle: l'intervalle de la moyenne glissante

@liste_moyennes: le jeu de valeur apres moyenne glissante 
/!\ une moyenne glissante retourne un tableau plus petit que celui en entré /!\
"""
def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes = [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
    
    
    
    
    
    
    
    
    
    
    
    