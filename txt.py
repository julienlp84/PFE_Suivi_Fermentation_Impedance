import os

# Obtenez le chemin absolu du répertoire du script
repertoire_script = os.path.dirname(os.path.abspath(__file__))

# Concaténez le nom du fichier texte au chemin du répertoire du script
chemin_fichier = os.path.join(repertoire_script, 'R1.txt')

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

# Affichez les données lues à partir du fichier
print("Chaque élément de la quatrième colonne de chaque ligne du fichier :")
print(donnees_fichier)





