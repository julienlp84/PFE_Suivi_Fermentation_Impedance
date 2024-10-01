# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:21:28 2021

@author: valentin.lefebvre01

plot_print.py: ensemble des fct necessaire a l'affichage en temps reel des graphes
"""

from datetime import datetime
#from math import log10

"""
renvoie la position de la frequence la plus proche de 40kHz
@param debut_log: la valeur en Hz de la frequence minimum
@param fin_log: la valeur en Hz de la frequence max
@param nb_point: le nombre de point qui compose notre echelle log
@param freq_voulu: la frequence de mesure pour laquelle on etudie la magnitude

@return pos_freq_mesure: la position dans le tableur de la frequence la plus proche de @freq_voulu
"""
def pt_mes_mag(debut_log,fin_log,nb_point,freq_voulu):
    r = fin_log/debut_log #ecart relatif entre les deux freq extremes
    rap_val = r**(1/(nb_point-1)) #rapport de valeurs qui correspond à chaque segment
    
    freq_mesure = debut_log
    pos_freq_mesure = 0 #position de la frequence de mesure
    while freq_mesure < freq_voulu:
        freq_mesure = freq_mesure*rap_val
        pos_freq_mesure = pos_freq_mesure + 1
    return pos_freq_mesure
        

"""
renvoie un tableau de la magnitude du liquide etudier 
en fonction d'une frequence precise ~40kHz
/!\ fonction a modifier pour pouvoir demander la frequence nous même /!\
    
recupere les données temporelles du fichier csv créé 
et la magnitude qui correspond a la frequence etudier

@param nom_fichier: le nom du fichier dans lequelle extraire les données
@param freq_analyse: la frequence utiliser pour analyser la magnitude en temporelle

@return tps: un tableau recuperant les données temporelle des mesure (abscisse)
@return Mag_graph: un tableau des magnitudes correspondantes aux données temporelle
@return Pha_graph: un tableau des phases correspondantes aux données temporelle
"""    
def tableau_tps_reel(nom_fichier, freq_analyse):
    f = open(nom_fichier,'r')
    data = f.read()
    data_split = data.split('\n')
    nb_ligne = len(data_split)
    
    nb_tram = nb_ligne//4
    Mag_graph = [0]*nb_tram
    Pha_graph = [0]*nb_tram
    tram_temps = ['']*nb_tram
    tram_freq = data_split[1].split(';')
    #print(tram_freq)
    nb_data = len(tram_freq) - 1 #il y a une donnée fantome a la fin du au dernier ;
    freq_debut = float(tram_freq[0])
    freq_fin = float(tram_freq[nb_data - 1])
    position_freq_mesure = pt_mes_mag(freq_debut, freq_fin, nb_data, freq_analyse)
    print(tram_freq[position_freq_mesure] + 'Hz')
    
    for li in range(nb_tram):
        tps_ligne = li*4 #numero de ligne ou on trouve une serie de data
        
        
        ligne_mag = data_split[tps_ligne + 2] #ligne avec la magnitude
        ligne_pha = data_split[tps_ligne + 3] #ligne avec la phase
        tram_temps[li] = data_split[tps_ligne]#ligne avec le relevé temporelle
        
        donnee_mag = ligne_mag.split(';')
        donnee_pha = ligne_pha.split(';')
        Mag_val = donnee_mag[position_freq_mesure]
        Pha_val = donnee_pha[position_freq_mesure]
        Mag_graph[li] = float(Mag_val)
        Pha_graph[li] = float(Pha_val)
    
    
    tps = conv_date_temps(tram_temps)
    f.close()
    #print(tps)
    #print(Mag_graph)
    return tps, Mag_graph, Pha_graph


"""
separe le tableau contenant toute les valeurs du dernier balayage en frequence
en 3 tableau de float contenant chacun un type de données differentes
@param tab: les données du dernier balayages en frequence effectué

@return freq: les frequences du dernier balayage
@return mag: les magnitude du dernier balayage
@return pha: les phases du dernier balayage
"""
def tableau_log(tab):
    
    nb_pt = len(tab[0])
    freq = [0]*nb_pt
    mag = [0]*nb_pt
    pha = [0]*nb_pt
    #print(tab_csv[0])
    for i in range(len(tab[0])):
        freq[i] = float(tab[0][i])
        #mag[i] = 20*log10(float(tab[1][i]))
        mag[i] = float(tab[1][i])
        pha[i] = float(tab[2][i])
    #print(x)
    #print(y)
    return freq,mag,pha

    
"""
convertie le temps d'analyse de toute les mesure
en un tableau de valeur en seconde

ce tableau est utiliser dans la fonction tableau_tps_reel()
pour recuperer les valeurs de temps en abscisse

@param tram_temps: un tableau de str de la forme "jj-mm-aaaa hh:mm:ss"

@return delta_seconde: un tableau contenant le decalage de temps entre 
                        la premiere mesure et les suivantes en seconde
"""    
def conv_date_temps(tram_temps):
    data = len(tram_temps)
    date_heure = ['']*data
    my_date = ['']*data
    heure = ['']*data
    date_now = ['']*data
    delta_seconde = [0.0]*data
    
    for i in range(data):
        date_heure[i] = tram_temps[i].split()
        
        #print(date_heure)
        my_date[i] = date_heure[i][0].split('-')
        #print(date)
        heure[i] = date_heure[i][1].split(':')
        #print(heure)
        
    date_origine = datetime(int(my_date[0][2]),int(my_date[0][1]),int(my_date[0][0]),int(heure[0][0]),int(heure[0][1]),int(heure[0][2]))
    for j in range(data):
        date_now[j] = datetime(int(my_date[j][2]),int(my_date[j][1]),int(my_date[j][0]),int(heure[j][0]),int(heure[j][1]),int(heure[j][2]))
        delta_seconde[j] = datetime.timestamp(date_now[j]) - datetime.timestamp(date_origine)
    return delta_seconde