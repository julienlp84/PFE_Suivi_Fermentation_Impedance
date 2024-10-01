import csv
import math
import matplotlib.pyplot as plt

def lire_csv(nom_fichier):
    tableau = []
    fichier = nom_fichier + '.csv'
    with open(fichier, newline='', encoding='utf-8-sig') as fichier_csv:
        
        
        lecteur_csv = csv.reader(fichier_csv)

        # Parcourir chaque ligne du fichier CSV
        for ligne in lecteur_csv:
            # Assurez-vous qu'il y a au moins une colonne avant d'essayer d'accéder à la première colonne
            if ligne:
                valeur_premiere_colonne = float(ligne[0])  # Convertir la valeur en float
                tableau.append(valeur_premiere_colonne)

    return tableau

def logarithme(impedances):
    logimp = []
    
    for imp in impedances:
        logimp.append(round(math.log10(imp),4))
    return logimp

def admittance(impedances):
    adm = []
    
    for imp in impedances:
        adm.append(0.000001*(round(100000/(imp),3)))
    return adm


###################### plot ##########################
def plot_points(c_al, imp, logimp, admittances):
    # Tracer les points des impédances
    plt.plot(c_al, imp, label='Impédances', marker='o', color = 'red')

    # Tracer les points des logarithmes des impédances
    plt.plot(c_al, logimp, label='Logarithme des impédances', marker='o', color='green')

    # Tracer les points des admittances
    plt.plot(c_al, admittances, label='Admittances', marker='o', color ='blue')

    # Ajouter des étiquettes aux axes et une légende
    plt.xlabel('Valeurs de c_al')
    plt.ylabel('Valeurs')
    plt.legend()

    # Afficher le graphique
    plt.show()
    
#######################################################
def main():
    nom_fichier_csv = input('Nom du fichier (sans extension): ')
    impedances = lire_csv(nom_fichier_csv)
    logimp = logarithme(impedances)
    admittances = admittance(impedances)
    
    c_al = [0,30,60,90,120,140]
    # Affichage ############################################
    
    print('\nImpédances :', impedances)
    print('\nLogarithme des impédances :', logimp)
    print('\nAdmittances :', admittances)
    plot_points(c_al, impedances, logimp, admittances)

if __name__ == "__main__":  
    main()
