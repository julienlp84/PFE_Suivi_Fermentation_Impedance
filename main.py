"""
@author: valentin.lefebvre01

installer sur un cmd wheel pythonnet et pyserial
commande : pip install wheel
voir librairie openpyxl

utilisé sur spyder (anaconda 3) python 3.8 pour voir le graphe en temps réel

"""
import matplotlib.pyplot as plt
from uart_to_csv import reading_serial , csv_write, user_init, STM_ComPort
from plot_print import tableau_tps_reel, tableau_log
from correction import graph_corrige
from time import gmtime, strftime
from res_fct_temps import res_fct_freq

"""
/!\ modifier le programme pour recuperer le port com de la stm automatiquement /!\
"""

temps = strftime("%d-%m-%Y_%Hh%Mm%Ss", gmtime()) #temps de la creation du fichier csv
nom_fichier = str(input('Nom fichier:')) + '.csv' #nom du fichier csv avec la date et l'heure de creation

#a = int(input('Choix du ComPort : '))
a = 2
com = STM_ComPort(a)


assert not(com == 'erreur') , 'Connexion au port com echoué. Essayer de reconnecter la carte'

plt.ion()


fig, (mag_fct_tps, mag_fct_fre, pha_fct_fre) = plt.subplots(nrows= 3, figsize=(20,16))


[nb_mesure, nb_ptn] = user_init(com)   
mesure = 0
freq_analyse = 1000 #frequence d'analyse pour le tableau temps réel
while mesure < nb_mesure:
  
    
    print('starting mesurement number ' + str(mesure + 1) + '\n')
    
    tab_csv = reading_serial(nb_ptn,com)
    mesure = mesure + 1
    
    if( tab_csv[1][0] != 0):
        csv_write(tab_csv, nom_fichier)
        
    
    
    
    pha_fct_fre.cla()
    mag_fct_fre.cla()
    mag_fct_tps.cla()
    pha_fct_fre.grid(which = 'both', axis= 'both')
    mag_fct_fre.grid(which = 'both', axis= 'both')
    mag_fct_tps.grid(which = 'both', axis= 'both')
    mag_fct_tps.set_xlabel("temps en seconde")
    mag_fct_tps.set_ylabel("magnitude en Ohm à " + str(freq_analyse) + "Hz")
    mag_fct_fre.set_xlabel("frequence en Hz")
    mag_fct_fre.set_ylabel("magnitude en Ohm")
    mag_fct_fre.set_xscale('log')
    pha_fct_fre.set_xlabel("frequence en Hz")
    pha_fct_fre.set_ylabel("phase en degre")
    pha_fct_fre.set_xscale('log')
    
    print('\nAt frequency:')
    [tps, Mag_graph, Pha_graph] = tableau_tps_reel(nom_fichier, freq_analyse)
    mag_fct_tps.plot(tps, Mag_graph)
    
    [fre, mag, pha] = tableau_log(tab_csv)
    print('reading : ' + str(Mag_graph[len(Mag_graph)-1]) + 'ohm')
    time = tps[len(tps)-1]
    h = int(time//3600)
    minute = int((time - h*3600)//60)
    seconde = int(time - h*3600 - minute*60)
    print('time: ' + str(h) + ':' + str(minute) + ':' + str(seconde) + '\n')
    mag_fct_fre.plot(fre, mag)
    pha_fct_fre.plot(fre, pha)
    
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
       
#res_fct_freq(nom_fichier) # a commenter si le pcb a besoin de correction
       
#graph_corrige(nom_fichier) # a commenter si le pcb n'a pas besoin de correction
#res_fct_freq('correction_' + nom_fichier) # a commenter si le pcb n'a pas besoin de correction


  

