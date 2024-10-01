# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:39:05 2021

@author: valentin.lefebvre01

uart_to_csv.py : ensemble des fonctions liant le programme main.py au port serie
"""

from time import gmtime, strftime
import serial
import serial.tools.list_ports

"""
recherche tout les portCOM comportant 'STM' dedans puis renvoie le numero de ce portCOM
on peut donc connecter la stm a nimporte quelle pc windows sans rechercher le portCOM

@return erreur: le programme n'a pas reussi a trouver de portCOM en lien avec une stm
@return ind[0]: le numero du port com relier a la carte stm

"""
def STM_ComPort(a):
    comlist = serial.tools.list_ports.comports()
    for i in range(len(comlist)):
        if 'STM' in str(comlist[i]):
            ind = str(comlist[a]).split()
            print('connexion sur le port : ' + ind[0])
            return ind[0]
    return 'erreur'
        
# def STM_ComPort2():
#     comlist = serial.tools.list_ports.comports()
#     for i in range(len(comlist)):
#         if 'STM' in str(comlist[i]):
#             ind = str(comlist[1]).split()
#             print('connexion sur le port : ' + ind[0])
#             return ind[0]
#     return 'erreur'

"""
fonction d'initialisation de l'appareil 
demande a l'utilisateur toute les infos necessaire concernant
la durée des mesures, la gamme de frequence et le nombre de mesure

@param com: le portCOM sur lequel on vient se connecter

@return nb_mesure : le nombre de mesure a effectuer
@return nb_ptn : le nombre de point durant le balayage en frequence
"""
def user_init(com):
    ser = serial.Serial(com, 230400, timeout=1) #port serie de la stm32
    #tps_mesure = int(input('temps entre deux mesures en seconde (max 9999s): '))
    tps_mesure = 8
    
    #print('une mesure sera effectuée toutes les ' + str(tps_mesure) + ' seconde\n')
    assert tps_mesure < 10000 and tps_mesure > 0 , 'le temps dois etre un entier > 0 et < 9999'
    
    #nb_mesure = int(input('nombre de mesure a effectuer : '))
    nb_mesure = 10
    
    #print('vous avez selectionné ' + str(nb_mesure) + ' mesure\n') #nombre de mesure à effectuer 
    assert nb_mesure > 0 , 'le nombre de mesure doit etre un nombre entier > 0'
    
    #freq_min = int(input('frequence minimal en Hz (min 300): '))
    freq_min = 300
    
    #assert freq_min>=300 , 'la frequence doit etre un nombre entier > 300'
    #freq_max = int(input('frequence maximal en Hz (max 199000): '))
    freq_max = 190000
    
    #assert freq_max<=199000 and freq_max>freq_min , 'la frequence max doit etre < 199000 et > à la frequence min'
    #nb_ptn = int(input('nombre de point lors du balayage en frequence (max 100):'))
    nb_ptn = 100
    
    assert nb_ptn <= 200 and nb_ptn > 0 , 'le nombre de point doit etre un nombre entier comprit entre 1 et 200'
    
    print('l\'ananyseur effectuera un balayage entre ' + str(freq_min) + 'Hz et ' + str(freq_max) + 'Hz pour un total de ' + str(nb_ptn) + ' points.' )
    tram = str(tps_mesure) + ';' + str(nb_ptn) + ';' + str(freq_min) + ';' + str(freq_max) + ';' + str(nb_mesure) + '.'
    if(len(tram)>29):
        print('erreur de saisie dans les valeurs')
        nb_mesure = user_init()
    else:
        tram_uart = tram + '*'*(29-len(tram))
        b = tram_uart.encode('utf-8')
        ser.write(b)
        ser.close()
        return nb_mesure,nb_ptn

"""
decode les trames qui arrive de la stm32 via le port serie 
les trames sont de la forme <frequence>;<magnitude>;<phase>

@param ligne: une ligne de donnée arrivant de la stm

@return [freq,Res,Pha]: un tableau contenant les données contenue dans @ligne
"""
def decodeur(ligne): 
    a1 = ligne.index(";")
    a2 = ligne.index(";", a1+1)
    freq = ligne[:a1]
    Res = ligne[a1+1:a2]
    Pha = ligne[a2+1:]
    return [freq,Res,Pha]


"""
lit les données de la carte arrivant sur le port serie et les stock dans un tableau

@param nb_ptn: le nombre de point lors du balayage en frequence
@param com: le portCOM sur lequel on vient se connecter

@return tab_csv: un tableau de 3 lignes et @nb_ptn colones avec:
    la frequence
    la magnitude
    la phase
"""
def reading_serial(nb_ptn,com):
    ser = serial.Serial(com, 230400, timeout=1) #port serie de la stm32 
    tab_csv = [[0]*nb_ptn,[0]*nb_ptn,[0]*nb_ptn]
    n_ligne = ''
    i = 0
    print('terminal reading: ...')
    while not(n_ligne == 'new'):
        donnee=ser.readline()
        if donnee:
          line = str(donnee)
          n_ligne = line[2:-5]
      
    while not(n_ligne == 'stop'):
      donnee=ser.readline()
      if donnee:
        
        line = str(donnee)
        n_ligne = line[2:-5]
        
        
        if not(n_ligne == 'stop'):
            tab1 = decodeur(n_ligne)
            
            tab_csv[0][i] = tab1[0]
            tab_csv[1][i] = tab1[1]
            tab_csv[2][i] = tab1[2]
            i = i+1
    
    for i in range(len(tab_csv[2])):
        if float(tab_csv[2][i]) > 180:
            tab_csv[2][i] = str(float(tab_csv[2][i]) - 360)
        if float(tab_csv[2][i]) < -180:
            tab_csv[2][i] = str(float(tab_csv[2][i]) + 360)
    ser.close()
    print('data received successfully')
    return tab_csv


"""
ecrit dans un fichier csv les trames qui arrive du stm32
tab -> tableau de donné avec :
    1er ligne la frequence
    2e ligne la magnitude
    3e ligne la phase

nom_fichier -> le nom du fichier sur lequel ecrire nos donnee au format csv

le fichier obtenue est organisé par paquet de 4 lignes avec:
    1er ligne la date de la mesure
    2e ligne la frequence
    3e ligne la magnitude
    4e ligne la phase
    
à chaque nouveau paquet de donné 4 nouvelles lignes sont ecrite à la suite

@param tab: le tableau de valeur contenant les données à ecrire
@param nom_fichier: le nom du fichier sur lequel ecrire les données
"""
def csv_write(tab,nom_fichier):
    f = open(nom_fichier, 'a')
    temps = strftime("%d-%m-%Y %H:%M:%S", gmtime())
    f.write(temps + "\n")
    for i in range(len(tab[0])):
        f.write(str(tab[0][i]) + ';')
    f.write("\n")
    
    for j in range(len(tab[1])):
        f.write(str(tab[1][j]) + ';')
    f.write("\n")
    
    for k in range(len(tab[2])):
        f.write(str(tab[2][k]) + ';')
    f.write("\n")
    f.close()
    print('data stored successfully in ' + nom_fichier)