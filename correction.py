# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:12:07 2022

@author: valentin.lefebvre01

correction.py: ensemble des fct necessaire a la correction du pcb "impedance analizer v1"

/!\ en cas de creation d'un nouveau pcb ce prgramme devient obsolette
puisqu'il ne corrige que les erreurs recurrante induite par le pcb v1 /!\
"""

"""
partie de fct de la correction de la valeur d'impedance lu du au pcb impedance analizer v1

@param R: la magnitude obtenu sur l'analyseur qui necessite une correction

@return reajustage: le resultat de la fct de reajustage
"""
def f_de_R(R):
    err_rel_moy = [[100,218,324,467,679,977,2180,3289,4680,6770,8180,9979,14961],[-3.1512495,-3.607755995,-2.779593642,-2.951666949,-2.740084113,-1.734501356,-0.862043913,0.484583855,0.417881484,0.457937424,0.667535332,0.851196957,1.091659456]]
    
    if (R>=100 and R<14961): 
        i=0
        while(R>err_rel_moy[0][i]):
            i = i+1
        i = i-1
        pourcentage_res_inf = 1- (R - err_rel_moy[0][i])/(err_rel_moy[0][i+1]-err_rel_moy[0][i])
        pourcentage_res_sup = 1- (err_rel_moy[0][i + 1] - R)/(err_rel_moy[0][i+1]-err_rel_moy[0][i])
        reajustage = pourcentage_res_inf*err_rel_moy[1][i] + pourcentage_res_sup*err_rel_moy[1][i+1]
    elif R>= 14961:
        reajustage = err_rel_moy[1][12]
    else:
        reajustage = err_rel_moy[1][0]
        
    return reajustage

"""
fct de correction des valeurs d'impedance obtenu lors d'un balayage en frequence par impedance analizer v1

@param Robt: la magnitude obtenu sur l'analyseur qui necessite une correction
@param freq: la frequence a l'aquelle on corrige notre magnitude Robt

@return Rcorr: la magnitude corrigée a la frequence freq
"""
def correction(Robt,freq):
    #erreur relative moyenne produite par le pcb de l'ad5941 version 1 en fct de la frequence
    err_rel_moy = [[300,320.0199,341.0758,364.077,389.033,415.0544,443.0524,473.0387,505.0261,539.0281,575.0591,614.0346,655.0711,699.0861,746.0984,797.028,850.0962,908.0258,969.0413,1034.0685,1104.0352,1178.0709,1258.0074,1342.0782,1433.0193,1529.0692,1632.0689,1742.062,1859.0953,1985.0187,2118.0852,2261.0518,2413.0789,2576.0313,2749.078,2934.0927,3132.054,3343.0458,3568.0578,3808.0856,4065.0312,4339.0035,4631.0189,4943.0014,5275.0834,5631.0064,6010.0212,6414.0888,6846.0812,7307.0818,7799.0864,8325.0041,8885.0578,9483.0857,10122.0419,10803.0977,11531.0424,12307.0852,13136.0558,14021.0062,14965.0121,15972.0745,17048.0212,18196.0095,19421.0265,20728.0927,22124.0636,23614.032,25204.0308,26901.0351,28712.0658,30645.0923,32709.0359,34911.073,37262.0386,39771.032,42449.0179,45307.0347,48357.0957,51613.0968,55089.0214,58798.0453,62757.0437,66982.0992,71493.0054,76306.0781,81444.0632,86928.0421,92781.0437,99028.0546,105696.0289,112812.0976,120408.0835,128516.014,137169.0328,146405.014,156262.0812,166784.0218,178014.0046,190000,200000]
,[-2.773480692,-2.298285128,-2.736649217,-2.190486431,-1.500796124,-1.528761839,-1.026346921,-0.713075476,-1.03937535,-0.91029989,-0.932191903,-0.955960427,-0.638077035,-0.547491179,-0.928319912,-0.515824093,-0.689231264,-1.095106711,-0.965122667,-1.234448519,-1.178077903,-1.099166736,-1.073665347,-1.060935059,-0.823564297,-0.50488581,-0.494132453,-0.659272277,-0.720675005,-0.742560958,-0.698251069,-0.785226048,-0.757002851,-0.912501046,-0.777875639,-0.772793887,-0.816686622,-0.926545444,-1.114432835,-1.13598685,-1.003808636,-1.248486739,-1.197207553,-1.289262054,-1.375815397,-1.647923059,-1.412919603,-1.717322463,-1.607328365,-1.719066952,-1.918667281,-1.999077435,-2.002624614,-2.164960815,-2.32121383,-2.40180388,-2.56421876,-2.630638164,-2.83554682,-2.947219519,-3.078139335,-3.303861396,-3.314038334,-3.752081654,-3.924218253,-4.147945849,-4.169295315,-4.149280703,-4.071199478,-3.893590702,-3.413008711,-2.538747574,-1.824024748,-1.114251533,-0.252960248,0.475909033,0.969820948,0.897952117,0.706624713,-3.855029558,-3.507476611,-2.566356953,-1.71267987,-0.945449639,-0.059961612,0.71073461,1.219370222,1.271773477,1.038010477,0.601760568,-3.671451065,-3.671978665,-3.443199686,-3.323772173,-2.361896562,-1.653922393,-0.770766372,0.143665842,1.044362212,1.686104267,1.686104267]]
    #erreur relative moyenne produite par le pcb de l'ad5941 version 1
    err_moy = -1.480031774
    i=0
    assert freq >= 300 , 'frequence trop basse doit etre superieur a 300Hz '
    assert freq < 200000 , 'frequence trop elevé doit etre strictement inferieur a 200kHz '
    while(freq>err_rel_moy[0][i]):
        i = i+1
    

    pourcentage_freq_inf = 1- (freq - err_rel_moy[0][i])/(err_rel_moy[0][i+1]-err_rel_moy[0][i])
    pourcentage_freq_sup = 1- (err_rel_moy[0][i + 1] - freq)/(err_rel_moy[0][i+1]-err_rel_moy[0][i])
    err_rel_moy_Robt =  pourcentage_freq_inf*err_rel_moy[1][i] + pourcentage_freq_sup*err_rel_moy[1][i+1]  
    
    
    Rcorr = Robt*(1+ (err_rel_moy_Robt + f_de_R(Robt) - err_moy)/100)
    return Rcorr

"""
fct de correction d'un ensemble de mesure stocké dans un fichier .csv comme dans la fct csv_write (uart_to_csv.py)

@param nom_fichier: le fichier .csv obtenu apres une serie de mesure faite a l'aide du impedance analizer v1
"""
def graph_corrige(nom_fichier):
    f = open(nom_fichier,'r')
    print('reading file' + nom_fichier)
    data = f.read()
    data_split = data.split('\n')
    nb_ligne = len(data_split)
    tram_freq = data_split[1].split(';')
    tram_freq = tram_freq[:-1]
    #print(tram_freq)
    
    nb_data = len(tram_freq)
    nb_tram = nb_ligne//4
    mag = [[0]*nb_data for i in range(nb_tram)]
    mag_corr = [[0]*nb_data for i in range(nb_tram)]
    pha = ['']*nb_tram
    date = ['']*nb_tram
    avancee = 0
    print('done')
    print('data corrections')
    for li in range(nb_tram):
        mag[li] = data_split[li*4 + 2].split(';')
        pha[li] = data_split[li*4 + 3]
        date[li] = data_split[li*4]
        
        avancee = avancement(li, nb_tram, avancee)
        
        for i in range(nb_data):
            mag_corr[li][i] = correction(float(mag[li][i]),float(tram_freq[i]))
    print('done')
    f.close()
    #print(mag_corr)
    f = open('correction_' + nom_fichier, 'a')
    
    print('creating file correction_' + nom_fichier)
    for ligne in range(nb_tram):
        f.write(date[ligne] + '\n')
        f.write(data_split[1] + '\n')
        for j in range(len(mag_corr[ligne])):
            
            
            f.write(str(mag_corr[ligne][j]) + ';')
        f.write("\n")
        f.write(pha[ligne] + '\n')
        avancee = avancement(ligne, nb_tram, avancee)

    f.close()
    print('correction_' + nom_fichier + ' has successfully been built')
    
"""
fct qui affiche l'avancement d'une fonction qui boucle

@param ind: l'indice actuelle de la boucle
@param maximum: l'indice maximum atteint par la boucle
@param avancee: l'avancee precedente du programme "initialiser a 0 lors de la premiere utilisation"
"""
def avancement(ind, maximum, avancee):
    if ind>maximum/5 and avancee == 0:
        print('20%')
        avancee = 1
    elif ind>2*maximum/5 and avancee == 1:
        print('40%')
        avancee = 2
    elif ind>3*maximum/5 and avancee == 2:
        print('60%')
        avancee = 3
    elif ind>4*maximum/5 and avancee == 3:
        print('80%')
        avancee = 4
    elif ind == maximum - 1 and avancee == 4:
        print('100%')
        avancee = 0      
    return avancee
    
