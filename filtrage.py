# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:00:22 2024

@author: BAT14
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import cmath
import math

def impedance_complex(module, phase_deg):
    # Utilisation de la fonction rect pour obtenir les coordonnées cartésiennes
    phase_rad = math.radians(phase_deg)
    impedance_complexe =cmath.rect(module, phase_rad)
    
    return impedance_complexe

def retrouver_phase_magnitude(impedance):
    # Calculer la phase en utilisant la fonction arctangente
    phase_radians = cmath.phase(impedance)
    
    # Convertir la phase en degrés
    phase_degrees = math.degrees(phase_radians)
    
    # Récupérer la magnitude du nombre complexe
    magnitude = abs(impedance)
    
    return magnitude, phase_degrees

# Exemple d'utilisation
magnitude = 407  # Remplacez par la grandeur de magnitude souhaitée
phase = 161      # Remplacez par la grandeur de phase souhaitée

impedance_complexe = impedance_complex(magnitude, phase)
phase_retrouvee,magnitude_retrouvee = retrouver_phase_magnitude(impedance_complexe)

print(f"Impédance complexe : {impedance_complexe}")
print(f"Magnitude retrouvée : {magnitude_retrouvee}")
print(f"Phase retrouvée : {phase_retrouvee} degrés")


