# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:20:51 2024

@author: Matthieu Perrin
@EN_Translate
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import cmath
import math
import csv
import chardet

#################################################################################################   
#                                           USEFUL FUNCTIONS                                    #
#################################################################################################
def round_significative(a):
    multiplier = 0
    if a<0 and(a>-1):
        while (a>-1):
            a=a*10 
            print("a process:",a)
            multiplier+=1
    elif a>0 and (a<1):
        while (a<1):
            a=a*10 
            print("a process:",a)
            multiplier+=1
            
    a = round(a,2)
    print("a_round =", a)
    print("multiplier: ", multiplier)
    a = a* pow(10,-multiplier)
    return a

def string_to_float(table):
    converted_values = []

    for cell in table:
        # Ignore empty strings
        if cell.strip() != '':
            # Convert to float with 2 decimal places
            converted_value = round(float(cell), 2)
            converted_values.append(converted_value)

    return converted_values

def sort_ascending(table):
    n = len(table)

    for i in range(n - 1):
        # Find the minimum element in the remaining table
        min_index = i
        for j in range(i + 1, n):
            if table[j] < table[min_index]:
                min_index = j

        # Swap the minimum element with the current position element
        table[i], table[min_index] = table[min_index], table[i]
        
def sort_descending(table):
    
    n = len(table)

    for i in range(n - 1):
        # Find the minimum element in the remaining table
        min_index = i
        for j in range(i + 1, n):
            if table[j] > table[min_index]:
                min_index = j

        # Swap the minimum element with the current position element
        table[i], table[min_index] = table[min_index], table[i]  

def get_file_data(file):
    # Detect the file encoding
    with open(file, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    
    encoding = result['encoding']

    # Read data from CSV file using the detected encoding
    data = []
    try:
        with open(file, newline='', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                data.append(row)
    except UnicodeDecodeError:
        # If the detected encoding fails, try UTF-8
        try:
            with open(file, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    data.append(row)
        except UnicodeDecodeError:
            # If UTF-8 fails, try ISO-8859-1
            with open(file, newline='', encoding='iso-8859-1') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row in reader:
                    data.append(row)

    return data

def average_impedance_file(file): # calculate the average magnitude of all measurements in a file at each frequency
    # Get file data  
    file_data = get_file_data(file)
    #print( "file data: ",file_data)
    # Check if data was retrieved
    if file_data is None:
        return None

    # List to store averages
    averages = []

    # Rows to average
    rows_to_average = []
    
    frequencies=string_to_float(file_data[1])
    
    num_rows=len(file_data)
    num_measurements=int(num_rows/4)
    num_columns=len(file_data[1])
    #print("file: ", file)
    #print("num measurements", num_measurements)
    for i in range(num_measurements):
        rows_to_average.append(2+4*i)
    #print('rows to average',rows_to_average)
    #print('num file rows :',num_rows)
    #print('frequencies:', frequencies)
    #print('num freq points (columns) : ', len(frequencies))
    
    for column in range(len(frequencies)):
        sum_imp=0
        for row in rows_to_average:
            sum_imp = sum_imp + round(float(file_data[row][column]),2)
            avg_col=round((sum_imp*4)/num_rows,1)
        averages.append(avg_col)
    
    #print('num averages:', len(averages))
    return averages, frequencies

def impedance_at_1_freq(file, freq): # calculate the average magnitude of all measurements in a file at a specific frequency
    # Get file data  
    file_data = get_file_data(file)
    #print( "file data: ",file_data)
    # Check if data was retrieved
    if file_data is None:
        return None

    # List to store impedances
    impedances = []

    # Rows to average
    rows_to_average = []
    
    frequencies=string_to_float(file_data[1])
    freq_index = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - freq))
    freq_value = frequencies[freq_index]
    num_rows=len(file_data)
    num_measurements=int(num_rows/4)
    num_columns=len(file_data[1])
    #print("file: ", file)
    #print("num measurements", num_measurements)
    for i in range(num_measurements):
        rows_to_average.append(2+4*i)
    #print('rows to average',rows_to_average)
    #print('num file rows :',num_rows)
    #print('frequencies:', frequencies)
    #print('num freq points (columns) : ', len(frequencies))
    
    for row in rows_to_average:
            imp =round(float(file_data[row][freq_index]),2)
            #print(imp)
            impedances.append(imp)
    
    #print('num averages:', len(averages))
    return impedances, freq_value

def impedance_at_all_freqs(file):
    # Get file data  
    file_data = get_file_data(file)
    
    # Check if data was retrieved
    if file_data is None:
        return None

    # List to store impedances for each frequency
    all_impedances = []
    
    # Extract frequencies from the second row (assuming it is stored there)
    frequencies = string_to_float(file_data[1])
    
    # Determine number of rows for averaging, assuming data layout from your previous description
    num_rows = len(file_data)
    num_measurements = int(num_rows / 4)  # This assumes every 4 rows belong to a new measurement set

    # Iterate over each frequency
    for freq_index in range(len(frequencies)):
        freq_value = frequencies[freq_index]
        impedances = []
        
        # Rows corresponding to measurements
        rows_to_average = [2 + 4 * i for i in range(num_measurements)]
        
        # Collect impedance data for the current frequency across all relevant rows
        for row in rows_to_average:
            imp = round(float(file_data[row][freq_index]), 2)
            impedances.append(imp)
        
        # Store the average impedance for the current frequency
        avg_impedance = sum(impedances) / len(impedances)
        all_impedances.append(avg_impedance)

    return all_impedances, frequencies

def average_phase_file(file): # calculate the average phase of all measurements at each frequency
    # Get file data
    file_data = get_file_data(file)
    #print( "file data: ",file_data)
    # Check if data was retrieved
    if file_data is None:
        return None

    # List to store averages
    averages = []

    # Rows to average
    rows_to_average = []
    
    frequencies=string_to_float(file_data[1])
    
    num_rows=len(file_data)
    num_measurements=int(num_rows/4)
    num_columns=len(file_data[1])
    print("num measurements", num_measurements)
    for i in range(num_measurements):
        rows_to_average.append(3+4*i)
    print('rows to average',rows_to_average)
    #print('num file rows :',num_rows)
    #print('frequencies:', frequencies)
    #print('num freq points (columns) : ', len(frequencies))
    
    for column in range(len(frequencies)):
        sum_imp=0
        for row in rows_to_average:
            sum_imp = sum_imp + round(float(file_data[row][column]),2)
            avg_col=round((sum_imp*4)/num_rows,1)
        averages.append(avg_col)
    
    #print('num averages:', len(averages))
    return averages, frequencies

#################################################################################################   
#                                           AVERAGES                                            #
#################################################################################################   
def folder_averages(folder):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    # Check if there is at least one CSV file in the folder
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    # Create a figure with two subplots (for impedance and phase)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Iterate over all CSV files in the folder
    for file in csv_files:
        file_path = os.path.join(folder, file)

        # Calculate averages for the current file (impedance and phase)
        avg_impedance, freq_impedance = average_impedance_file(file_path)
        avg_phase, freq_phase = average_phase_file(file_path)
        # Plot impedance on the first subplot
    axes[0].plot(freq_impedance, avg_impedance, linestyle='-', label=file)

    # Plot phase on the second subplot
    axes[1].plot(freq_phase, avg_phase, linestyle='-', label=file)

    # Configure axes and display legends
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Impedance Magnitude')
    axes[0].grid(True)
    axes[0].set_ylim(0, 2000)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Impedance Phase')
    axes[1].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plots
    plt.show()
    
def print_folder_averages(folder): # fills a table containing the average rows
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    # Check if there is at least one CSV file in the folder
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    # 2D table to store the averages of each file
    folder_averages = [[]]
    average_20khz=[]
    # Iterate over all CSV files in the folder
    plt.figure()
    
    for file in csv_files:
        
        file_path = os.path.join(folder, file)
        
        # Calculate averages for the current file
        file_avg, file_freq = average_impedance_file(file_path)
        #avg_phase, freq_phase = average_phase_file(file_path)
        # Use different colors with dotted lines if necessary
        magn_20khz = file_avg[62]
        average_20khz.append(magn_20khz)
        #line_style = '-' if len(file_avg) > 10 else '-'
        plt.plot(file_freq, file_avg, linestyle='-', label=file)
        folder_averages.append(file_avg)
        folder_averages[0].append(file)
        
        
        #print("values for csv ", folder_averages)
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim(0, 1000)  # Set y-axis limits
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Average Impedance (ohms)')
        #plt.title('Average Impedance of All Files vs Frequency (Log Scale)')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        #plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=len(csv_files))
        plt.show()
        
        
        return folder_averages, file_freq
#################################################################################################
#                                    IMPEDANCE VS CONCENTRATION
#################################################################################################

def csv_for_scilab(name, folder_averages, frequencies): ##### to get the averages of each measurement in a folder for all frequencies, with the measurement name as the first column
    file_name = name + '.csv'
    print("begin filling csv")
    print(" data :", folder_averages)
    print(" x-axis :" , frequencies)
    try:
        with open(file_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            # Write frequencies in the first row
            #csvwriter.writerow(frequencies + [frequencies[0]])
            csvwriter.writerow( [''] + frequencies)

            # Write data starting from the third row
            for i, data_row in enumerate(folder_averages[1:], start=1):
                # Check if the index is within the range of folder_averages[0]
                label = folder_averages[0][i-1] if i-1 < len(folder_averages[0]) else ''
                #csvwriter.writerow(data_row + [label])
                csvwriter.writerow([label] + data_row)

        print(f"Data successfully added to {file_name}.")
    except Exception as e:
        print(f"An error occurred while writing CSV data: {e}")
        
def imp_points_freq_for_conc(folder, param):
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    averages, frequencies = folder_averages(folder)
    print('averages:', averages)

    freq_indices = [19, 54, 90]
    freq = [frequencies[i] for i in freq_indices]
    
    print('\n----- Impedance vs Concentration at 3 Frequencies (1k, 10k, 100k) ------')
    print('studying these frequencies:', freq)
    print('averages: ', averages)
    points = [[] for _ in range(len(freq))]
    
    for j in range(len(points)):
        if len(param)==2:
            points[j].append(averages[1][freq_indices[j]])
            points[j].append(averages[2][freq_indices[j]])
        elif len(param)==3:
            points[j].append(averages[1][freq_indices[j]])
            points[j].append(averages[2][freq_indices[j]])
            points[j].append(averages[3][freq_indices[j]])
        sort_ascending(points[j])
    print('Impedance Values for Conditions:', points)
    return points

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

def R2(x_points, y_points, a, b):
    """
    Calculate R^2 for a linear model (y = ax + b).
    Parameters:
        - x_points: Array or list of x-coordinates.
        - y_points: Array or list of corresponding y-coordinates.
        - a: Slope of the linear model.
        - b: Intercept of the linear model.
        
        Returns:
            - r_squared: R^2 value indicating the goodness of fit.
            """
    n = len(x_points)
            
    # Calculate mean of y_points
    y_mean = sum(y_points) / n

    # Calculate predicted y values based on the linear model
    y_predicted = [a * x + b for x in x_points]
    
    # Calculate sums for R^2 calculation
    ss_total = sum((y - y_mean) ** 2 for y in y_points)
    ss_residual = sum((y - y_predicted[i]) ** 2 for i, y in enumerate(y_points))
    
    # Calculate R^2
    r_squared = 1 - (ss_residual / ss_total)
    
    return round(r_squared,3)


def impedance_vs_concentration(freq_to_study, mode):
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 6))
    
    #### CONCENTRATIONS TO STUDY
    #c_sugar = [60, 120, 180, 240, 300]
    c_sugar = [0,150,300]
    
    #c_alcohol = [0, 30, 60, 90, 120, 140]
    c_alcohol = [0,71,141]
    
    c_nitrogen = [0, 0.25, 0.5]

    #### RETRIEVE IMPEDANCE DATA FROM CSV MEASUREMENT FILES
    monodatapath = 'C:/Users/chiu-wing-timothy.ho/Downloads/Fermentation data and Bode analysis/IMPORTANT/MONO_PARAMETERS_MEAN/'
    multidatapath = 'C:/Users/user/Downloads/Fermentation data and Bode analysis/IMPORTANT/MULTI_PARAMETERS_MEAN/'
    #data_sugar = get_file_data(monodatapath+'moyennes_sucre.csv')
    data_sugar = get_file_data(multidatapath+'sugar_with_halfN&A.csv') # should correspond to conditions 8, 9 and 10 on experiment plan
   
    #data_alcohol = get_file_data(monodatapath+'moyennes_alcool.csv')
    data_alcohol = get_file_data(multidatapath+'alcohol_with_halfN&S.csv') # should correspond to conditions 8, 13 and 14 on experiment plan
    
    #data_nitrogen = get_file_data(monodatapath+'moyennes_azote.csv')
    data_nitrogen = get_file_data(multidatapath+'nitrogen_with_halfS&A.csv') # should correspond to conditions 8, 11 and 12 on experiment plan

    num_rows_sugar = len(c_sugar)
    num_rows_alcohol = len(c_alcohol)
    num_rows_nitrogen = len(c_nitrogen)
    print("Number of sugar concentrations:", num_rows_sugar)
    print("Number of alcohol concentrations:", num_rows_alcohol)
    print("Number of nitrogen concentrations:", num_rows_nitrogen)

    freq_indices = []
    imp_sugar = []
    imp_alcohol = []
    imp_nitrogen = []
    
    frequencies = string_to_float(data_alcohol[0])

    for freq in freq_to_study:
        index = index_closest_value(frequencies, freq)
        freq_indices.append(index)
    for m in range(len(freq_indices)):
        freq_indices[m] = freq_indices[m] + 2

    print("Frequencies to study:", freq_to_study)
    print("Indices of these frequencies:", freq_indices)
    print("Sugar impedance:", imp_sugar)

    for i, freq in enumerate(freq_indices):
        xy_sum = 0
        y_sum = 0
        x_sum = 0
        x2_sum = 0
        xbyb_sum = 0
        xb_sum = 0
        yb_sum = 0
        xb2_sum = 0
        xcyc_sum = 0
        xc_sum = 0
        yc_sum = 0
        xc2_sum = 0

        imp_sugar.append([])
        imp_alcohol.append([])
        imp_nitrogen.append([])
        
        if mode == "Impedance":
            unit = "Ω"
            for j in range(num_rows_sugar):
                imp_sugar[i].append((float(data_sugar[j + 1][freq])))
            for k in range(num_rows_alcohol):
                imp_alcohol[i].append((float(data_alcohol[k + 1][freq])))
            for l in range(num_rows_nitrogen):
                imp_nitrogen[i].append(((float(data_nitrogen[l + 1][freq]))))
                
            sort_ascending(imp_sugar[i])
            sort_ascending(imp_alcohol[i])
            sort_descending(imp_nitrogen[i])   
        
        elif mode == "Admittance":
            unit = "S"
            for j in range(num_rows_sugar):
                imp_sugar[i].append(1/(float(data_sugar[j + 1][freq])))
            for k in range(num_rows_alcohol):
                imp_alcohol[i].append(1/(float(data_alcohol[k + 1][freq])))
            for l in range(num_rows_nitrogen):
                imp_nitrogen[i].append(1/((float(data_nitrogen[l + 1][freq]))))
                
            sort_descending(imp_sugar[i])
            sort_descending(imp_alcohol[i])
            sort_ascending(imp_nitrogen[i])
        

        # ## Least squares method: model y=ax+b -----------------------
        y = imp_sugar[i]
        yb = imp_alcohol[i]
        yc = imp_nitrogen[i]

        x = c_sugar
        xb = c_alcohol
        xc = c_nitrogen

        for t in range(num_rows_sugar):
            x_sum += x[t]
            y_sum += y[t]
            xy_sum += y[t] * x[t]
            x2_sum += x[t] * x[t]
            #print("t:", t)

        for t2 in range(num_rows_alcohol):
            xb_sum += xb[t2]
            yb_sum += yb[t2]
            xbyb_sum += yb[t2] * xb[t2]
            xb2_sum += xb[t2] * xb[t2]
        
        for t3 in range(num_rows_nitrogen):
            xc_sum += xc[t3]
            yc_sum += yc[t3]
            xcyc_sum += yc[t3] * xc[t3]
            xc2_sum += xc[t3] * xc[t3]

        a = (num_rows_sugar * xy_sum - y_sum * x_sum) / (num_rows_sugar * x2_sum - np.square(x_sum))
        b = (y_sum * x2_sum - xy_sum * x_sum) / (num_rows_sugar * x2_sum - np.square(x_sum))
        
        a2 = (num_rows_alcohol * xbyb_sum - yb_sum * xb_sum) / (num_rows_alcohol * xb2_sum - np.square(xb_sum))
        b2 = (yb_sum * xb2_sum - xbyb_sum * xb_sum) / (num_rows_alcohol * xb2_sum - np.square(xb_sum))

        a3 = (num_rows_nitrogen * xcyc_sum - yc_sum * xc_sum) / (num_rows_nitrogen * xc2_sum - np.square(xc_sum))
        b3 = (yc_sum * xc2_sum - xcyc_sum * xc_sum) / (num_rows_nitrogen * xc2_sum - np.square(xc_sum))
        
        
        #R2 = 1 – (Sum from 1 to n of (y_i – ^y_i)²)/(Sum from 1 to n of (y_i – y_bar)²)
        R2_sugar = 1 - (y_sum)
        # cov_xy = np.cov(x, y)[0, 1]  # Covariance between x and y
        # var_x = np.var(x)  # Variance of x
        # abis = cov_xy / var_x
        # print("a with covariance:", abis)
        
        cov_sugar_alcohol = np.cov(x, xb, xc)
        print("Covariance between sugar and alcohol =", cov_sugar_alcohol)
        
        cov_sugar_nitrogen = np.cov(x, xc)
        print("Covariance between sugar and nitrogen =", cov_sugar_nitrogen)
        
        cov_alcohol_nitrogen = np.cov(xb, xc)
        print("Covariance between alcohol and nitrogen =", cov_alcohol_nitrogen)
        
        a = round_significative(a)
        b = round_significative(b)
        a2 = round_significative(a2)
        b2 = round_significative(b2)
        
        a3 = round_significative(a3)
        b3 = round_significative(b3)
        
        print("a_final =", a)
        print("b =", b)
        print("a2 =", a2)
        print("b2 =", b2)
        print("a3 =", a3)
        print("b3 =", b3)
        
        ### Print variations
        R2_sugar  = R2(c_sugar, imp_sugar[i], a, b)
        R2_alcohol = R2(c_alcohol, imp_alcohol[i], a2, b2)
        R2_nitrogen  = R2(c_nitrogen, imp_nitrogen[i], a3, b3)
        print(f"{mode} vs. sugar  : R²=", R2_sugar)
        print(f"{mode} vs. alcohol : R²=", R2_alcohol)
        print(f"{mode} vs. nitrogen : R²=", R2_nitrogen)
        
        axes[0].plot(c_sugar, imp_sugar[i], marker='o', linestyle='')#, label=f'sugar {freq_to_study[i]} Hz')
        axes[0].plot(c_sugar, [a * x + b for x in c_sugar], linestyle='-', color='green')#, label=f'Line: {a}x + {b}')
        axes[0].set_title(f' {mode} = {a}*c_sugar + {b} ({freq_to_study[i]/1000} kHz approximation)' , fontsize=10)
        axes[0].set_title(f' {mode} = {a}*c_sugar + {b} ' , fontsize=10)
        
        axes[1].plot(c_alcohol, imp_alcohol[i], marker='o', linestyle='')#, label=f'alcohol {freq_to_study[i]} Hz')
        axes[1].plot(c_alcohol, [a2 * o + b2 for o in c_alcohol], linestyle='-', color='green')#, label=f'Line: {a2}x + {b2}')
        axes[1].set_title(f' {mode} = {a2}*c_alcohol + {b2} ({freq_to_study[i]/1000} kHz approximation)', fontsize=10)
        axes[1].set_title(f' {mode} = {a2}*c_alcohol + {b2} ', fontsize=10)
        
        axes[2].plot(c_nitrogen, imp_nitrogen[i], marker='o', linestyle='')#, label=f'alcohol {freq_to_study[i]} Hz')
        axes[2].plot(c_nitrogen, [a3 * v + b3 for v in c_nitrogen], linestyle='-', color='green')#, label=f'Line: {a2}x + {b2}')
        axes[2].set_title(f' {mode} = {a3}*c_nitrogen + {b3} ({freq_to_study[i]/1000} kHz approximation)', fontsize=10)
        axes[2].set_title(f' {mode} = {a3}*c_nitrogen + {b3} ', fontsize=10)
        
        axes[0].set_xlabel('Sugar concentration (g/L)')
        axes[0].set_ylabel(f'{mode} ({unit})')
        axes[0].grid(True)
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
        
        axes[1].set_xlabel('Alcohol concentration (g/L)')
        axes[1].set_ylabel(f'{mode} ({unit})')
        axes[1].grid(True)
        
        axes[2].set_xlabel('Nitrogen concentration (g/L)')
        axes[2].set_ylabel(f'{mode} ({unit})')
        axes[2].grid(True)
        
        # Adjust the layout for better spacing
        plt.tight_layout()
        
        return imp_sugar, imp_alcohol, imp_nitrogen, c_sugar, c_alcohol, c_nitrogen
    
def impedance_vs_concentration1(freq_to_study, mode):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 8))

    #### CONCENTRATIONS TO STUDY
    c_sugar = [0, 150, 300]
    c_alcohol = [0, 71, 141]
    c_nitrogen = [0, 0.25, 0.5]

    #### RETRIEVE IMPEDANCE DATA FROM CSV MEASUREMENT FILES
    monodatapath = 'C:/Users/chiu-wing-timothy.ho/Downloads/Fermentation data and Bode analysis/IMPORTANT/MONO_PARAMETERS_MEAN/'
    multidatapath = 'C:/Users/chiu-wing-timothy.ho/Downloads/Fermentation data and Bode analysis/IMPORTANT/MULTI_PARAMETERS_MEAN/'
    
    data_sugar = get_file_data(multidatapath + 'sugar_with_halfN&A.csv')
    data_alcohol = get_file_data(multidatapath + 'alcohol_with_halfN&S.csv')
    data_nitrogen = get_file_data(multidatapath + 'nitrogen_with_halfS&A.csv')

    # Calculate average impedances for each file in the folder
    folder_avg, frequencies = folder_averages(multidatapath)

    freq_indices = [19, 54, 90]
    freq = [frequencies[0][i] for i in freq_indices]  # assuming frequencies are the same for all entries

    print('\n----- Impedance vs Concentration at 3 Frequencies (1k, 10k, 100k) ------')
    print('studying these frequencies:', freq)
    print('averages:', folder_avg)

    points = [[] for _ in range(len(freq))]

    for j in range(len(points)):
        for k in range(1, len(folder_avg)):
            points[j].append(folder_avg[k][freq_indices[j]])
        sort_ascending(points[j])

    print('Impedance Values for Conditions:', points)

    # Plot Impedance vs Concentration for each frequency
    for i, ax in enumerate(axes[:3]):
        if i == 0:
            ax.plot(c_sugar, points[i], marker='o', label=f'Freq: {freq[i]} Hz')
        elif i == 1:
            ax.plot(c_alcohol, points[i], marker='o', label=f'Freq: {freq[i]} Hz')
        elif i == 2:
            ax.plot(c_nitrogen, points[i], marker='o', label=f'Freq: {freq[i]} Hz')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Impedance (ohms)')
        ax.legend()
        ax.grid(True)

    # Prepare data for Nyquist plot
    real_parts = []
    imag_parts = []
    for k in range(1, len(folder_avg)):
        real_part = []
        imag_part = []
        for avg in folder_avg[k]:
            avg_complex = complex(avg, 0)  # assuming avg is purely real for simplicity
            real_part.append(avg_complex.real)
            imag_part.append(avg_complex.imag)
        real_parts.append(real_part)
        imag_parts.append(imag_part)

    # Plot Nyquist plot
    axes[3].set_title('Nyquist Plot')
    for i in range(len(real_parts)):
        axes[3].plot(real_parts[i], imag_parts[i], marker='o', label=f'Measurement {i + 1}')
    axes[3].set_xlabel('Real Part (ohms)')
    axes[3].set_ylabel('Imaginary Part (ohms)')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

    return points


    #################################################################################################   
#                                           IMP VS TIME                                         #
################################################################################################# 

def create_csv_file(imp, freq, file_name):
    # Check if the imp and freq lists have the same length
    file_name = file_name + ".csv"
    # Verify that the arrays have the same length
    if len(freq) != len(imp):
        raise ValueError("The freq and imp arrays must have the same length.")
    print("Number of measurements:", len(imp[0]))
    # Write the data to the CSV file
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')

        # Write the first row with headers
        headers = ['Frequencies'] + [f"measurement n°{j}" for j in range(len(imp[0])) for i in range(len(imp))]
        csvwriter.writerow(headers)

        # Write the data rows
        for i in range(len(freq)):
            row = [freq[i]] + [imp[i][j] for j in range(len(imp[i]))]
            csvwriter.writerow(row)
    print("Writing finished")

def index_closest_value(lst, value):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - value)) 

def duration_monitoring(file, freq_to_study):
    # Calculate the average magnitude of all measurements in a file for each frequency
    # Retrieve data from the file
    file_data = get_file_data(file)
    
    # Check if data was retrieved
    if file_data is None:
        return None

    # Impedance storage
    imp_magn = []
    imp_phase = []
    # Lines to study
    magnitude_lines = []
    phase_lines = []
    
    # Check if freq_to_study is a list
    if not isinstance(freq_to_study, list):
        freq_to_study = [freq_to_study]
        
    freq_indices = []
    frequencies = string_to_float(file_data[1])
    
    for freq in freq_to_study:
        index = index_closest_value(frequencies, freq)
        freq_indices.append(index)
        
        # 2D arrays
    for _ in freq_to_study:
        imp_magn.append([])
    for _ in freq_to_study:
        imp_phase.append([])
    
    num_lines = len(file_data)
    num_measurements = int(num_lines / 4)
    num_columns = len(file_data[1])

    for i in range(num_measurements):
        magnitude_lines.append(2 + 4 * i)
        phase_lines.append(3 + 4*i)

    a = 0
    for column in freq_indices:
        for line in magnitude_lines:
            imp_magn[a].append(round(float(file_data[line][column]), 2))
            imp_phase[a].append(round(float(file_data[line+1][column]), 2))
        a += 1
    
    return imp_magn, imp_phase, frequencies, freq_to_study

def display_derivative(imp_magn, frequencies, averaging_window):
    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)
    
    # Creation of the graph
    plt.figure(figsize=(10, 6))

    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calculation of the moving average
        moving_average = np.convolve(magnitudes, np.ones(averaging_window)/averaging_window, mode='valid')

        # Calculation of the numerical derivative
        derivative = np.gradient(moving_average, time[:len(moving_average)])

        # Display of the derivative with a different color for each frequency
        plt.plot(time[:len(moving_average)], derivative, marker='o', linestyle='-', label=f'Derivative - Frequency {frequencies[i]}')

    plt.title('Derivative of Points over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Derivative of Magnitudes')
    plt.legend()
    plt.grid(True)

def impedance_vs_hours(imp_magn, frequencies, frequencies_studied, averaging_window):
    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)
    
    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'black', 'pink']
    impedance_to_return = []
    # Creation of the graph
    plt.figure(figsize=(10, 6))
    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calculation of the moving average
        
        ############################ Type of filtering to apply ###########################################
        
        ### ------------------------------------------------------------------- No filtering 
        filtered_magnitudes = magnitudes
        
        ### ------------------------------------------------------------------- Moving average 
        moving_average = np.convolve(filtered_magnitudes, np.ones(averaging_window)/averaging_window, mode='valid')
        print("Moving average: ", moving_average)
        ### ------------------------------------------------------------------- Filter low points
        max_threshold = 30
        window_size = 3
        #filtered_magnitudes = filter_low_values(magnitudes, max_threshold, window_size)
        
        ###### To filter low points (next 4 lines)
        # max_threshold = 20
        # window_size = 10
        # filtered_magnitudes = filter_low_values(moving_average, max_threshold, window_size)
        # plt.plot(time[:len(filtered_magnitudes)], filtered_magnitudes, marker='o', markersize=1, linestyle='-', label=str(freq_khz) + ' kHz')

        # Display of the line with a different color for each frequency
        freq = round(frequencies_studied[i])
        freq_khz = round(freq/1000, 2)
        plt.plot(time[:len(moving_average)], moving_average, marker='o', markersize=2, linestyle='-', color=colors[i], label=str(freq_khz) + ' kHz')
        impedance_to_return.append(moving_average)
    plt.title(f'675 points: Impedance vs Time (Moving Average Window: {averaging_window} pts)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Moving Average Impedance (ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return impedance_to_return
def impedance_vs_seconds(imp_magn, frequencies, frequencies_studied):
    # Duration between each point in seconds
    duration_between_points = 8

    # Initialization of time
    current_time = timedelta(seconds=0)

    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm']

    # Creation of the graph
    plt.figure(figsize=(10, 6))

    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    for i, magnitudes in enumerate(imp_magn):
        # Display of the line with a different color for each frequency
        plt.plot(time[:len(magnitudes)], magnitudes, marker='o', linestyle='-', color=colors[i], label=frequencies[frequencies_studied[i]])

    plt.title('Impedance vs Time (Every 8 seconds)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Impedance (ohms)')
    plt.legend()
    plt.grid(True)
    plt.show()

def derivative_vs_hours(imp_magn, frequencies, frequencies_studied):
    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)
    
    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm']

    # Creation of the graph
    plt.figure(figsize=(10, 6))
    averaging_window = 1
    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    for i, magnitudes in enumerate(imp_magn):
        # Calculation of the moving average
        moving_average = np.convolve(magnitudes, np.ones(averaging_window)/averaging_window, mode='valid')
        freq = round(frequencies[frequencies_studied[i]])
        
        # Calculation of the derivative
        derivative = np.gradient(moving_average, time[:len(moving_average)])

        # Display of the derivative with a different color for each frequency
        plt.plot(time[:len(moving_average)], derivative, marker='o', linestyle='-', color=colors[i], label=freq)

    plt.title('Derivative of Points over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Derivative of Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.show()

def folder_vs_seconds(folder):
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    # Initialization of lists to store data
    imp_magn_folder = []
    frequencies_folder = []
    frequencies_studied_folder = []

    # Iterate over all CSV files in the folder
    for file in csv_files:
        # Check if the file name already ends with '.csv'
        if not file.endswith('.csv'):
            file = file + '.csv'
            
        file_path = os.path.join(folder, file)

        # Calculate data for the current file
        imp, freq, freq_studied = duration_monitoring(file_path)

        # Add the data of the current file to the lists
        imp_magn_folder.append(imp)
        frequencies_folder.append(freq)
        frequencies_studied_folder.append(freq_studied)

    # Create the graph for the measurements in the folder
    plt.figure(figsize=(12, 8))

    # Call the impedance_vs_seconds function for each file
    for i in range(len(csv_files)):
        impedance_vs_seconds(imp_magn_folder[i], frequencies_folder[i], frequencies_studied_folder[i])

    # Add labels and legends
    plt.title('Impedance vs Time for Each File (Every 8 seconds)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Impedance (ohms)')
    plt.legend(title='Files', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    
def variation_vs_hours(imp_magn, frequencies, frequencies_studied):
    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)

    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm']

    # Creation of the graph
    plt.figure(figsize=(10, 6))
    averaging_window = 40
    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    averaging_window = 1
    for i, magnitudes in enumerate(imp_magn):
        moving_average = np.convolve(magnitudes, np.ones(averaging_window) / averaging_window, mode='valid')
        # Calculation of the moving average
        first_point = moving_average[0]
        percentage_variation = [(point - first_point) / first_point * 100 for point in moving_average]

        # Display of the percentage variation curve with a different color for each frequency
        plt.plot(time[:len(moving_average)], percentage_variation, marker='o', linestyle='-', label=f'Variation - Frequency {frequencies[i]}')

    plt.title('Moving Average of Points over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Moving Average of Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.show()

def filter_low_values(magnitudes, max_threshold, window_size):
    """
    Filters the values by eliminating those below the maximum value minus 50 ohms in packets of 100 or 1000 values.

    Parameters:
    - magnitudes (list): List of magnitudes to filter.
    - max_threshold (int): Maximum threshold for filtering.
    - window_size (int): Size of the window for filtering.

    Returns:
    - filtered_magnitudes (list): List of filtered magnitudes.
    """

    filtered_magnitudes = []

    for i in range(0, len(magnitudes), window_size):
        window = magnitudes[i:i+window_size]

        # Apply the filter
        max_value = np.max(window)
        filtered_window = [val if val >= (max_value - 50) else np.nan for val in window]

        # Add the filtered values to the resulting list
        filtered_magnitudes.extend(filtered_window)

    return filtered_magnitudes

def variation_vs_hours(imp_magn, frequencies, frequencies_studied):
    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)

    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm']

    # Creation of the graph
    plt.figure(figsize=(10, 6))
    averaging_window = 40
    # Creation of the time list only once
    time = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    averaging_window = 1
    for i, magnitudes in enumerate(imp_magn):
        moving_average = np.convolve(magnitudes, np.ones(averaging_window) / averaging_window, mode='valid')
        # Calculation of the moving average
        first_point = moving_average[0]
        percentage_variation = [(point - first_point) / first_point * 100 for point in moving_average]

        # Display of the percentage variation curve with a different color for each frequency
        plt.plot(time[:len(moving_average)], percentage_variation, marker='o', linestyle='-', label=f'Variation - Frequency {frequencies[i]}')

    plt.title('Moving Average of Points over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Moving Average of Magnitudes')
    plt.legend()
    plt.grid(True)
    plt.show()

def filter_low_values(magnitudes, max_threshold, window_size):
    """
    Filters the values by eliminating those below the maximum value minus 50 ohms in packets of 100 or 1000 values.

    Parameters:
    - magnitudes (list): List of magnitudes to filter.
    - max_threshold (int): Maximum threshold for filtering.
    - window_size (int): Size of the window for filtering.

    Returns:
    - filtered_magnitudes (list): List of filtered magnitudes.
    """

    filtered_magnitudes = []

    for i in range(0, len(magnitudes), window_size):
        window = magnitudes[i:i+window_size]

        # Apply the filter
        max_value = np.max(window)
        filtered_window = [val if val >= (max_value - 50) else np.nan for val in window]

        # Add the filtered values to the resulting list
        filtered_magnitudes.extend(filtered_window)

    return filtered_magnitudes
########################## functions for filtering using complex numbers

def impedance_complex(module, phase_deg): ## convert magnitude and phase to complex impedance
    phase_rad = math.radians(phase_deg)
    impedance_complexe = cmath.rect(module, phase_rad)
    
    return impedance_complexe

def retrieve_phase_magnitude(impedance): ## retrieve magnitude and phase values
    phase_radians = cmath.phase(-impedance)
    phase_degrees = math.degrees(phase_radians)
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

def mean_complex_numbers(complexes):
    if not complexes:
        raise ValueError("The list of complex numbers must not be empty.")

    total = sum(complexes)
    mean = total / len(complexes)

    return mean

def mean_complex_file(file): # calculate the mean of magnitudes of all measurements in a file at each frequency
    # Retrieve data from the file
    file_data = get_file_data(file)
    
    # Check if data was retrieved
    if file_data is None:
        return None

    # List to store the means
    complex_data = []
    
    mean_magnitudes = []
    mean_phases = []

    # Lines to average
    magn_lines = []
    phase_lines = []
    
    frequencies = string_to_float(file_data[1])
    
    num_lines = len(file_data)
    num_measurements = int(num_lines / 4)
    num_columns = len(file_data[1])
    
    for i in range(num_measurements):
        magn_lines.append(2 + 4 * i)
        phase_lines.append(3 + 4 * i)
    
    for column in range(len(frequencies)): ## for each frequency
        sum_imp = 0
        for line in magn_lines:
            magn = round(float(file_data[line][column]), 2)
            phase = round(float(file_data[line + 1][column]), 2)
            complex_data.append(impedance_complex(magn, phase)) # list containing all complex numbers for 1 frequency
        mean_comp = mean_complex_numbers(complex_data) # calculate the mean of the list
        mean_magn, mean_phase = retrieve_phase_magnitude(mean_comp) # mean in non-complex form
        
        mean_magnitudes.append(mean_magn) # add the mean for each frequency
        mean_phases.append(mean_phase)
        
    return mean_magnitudes, mean_phases, frequencies
def means_by_complex_folder(folder):
    # Get the list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    means_magn = [[]]
    means_phase = [[]]

    # Check if there is at least one CSV file in the folder
    if not csv_files:
        print("No CSV files found in the folder.")
        return None

    # Create a figure with two subplots (for impedance and phase)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    # Iterate over all CSV files in the folder
    for file in csv_files:
        file_path = os.path.join(folder, file)

        # Calculate the means for the current file (impedance and phase)
        mean_impedance, mean_phase, frequencies = mean_complex_file(file_path)

        # Plot impedance on the first subplot
        axes[0].plot(frequencies, mean_impedance, linestyle='-', label=file)

        # Plot phase on the second subplot
        axes[1].plot(frequencies, mean_phase, linestyle='-', label=file)

        means_magn.append(mean_impedance)
        means_magn[0].append(file)
        means_phase.append(mean_phase)

    # Configure axes and display legends
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Impedance Magnitude (Ω)')
    axes[0].grid(True)
    axes[0].set_ylim(0, 1100)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Impedance Phase (degrees)')
    axes[1].grid(True)
    axes[1].set_ylim(-90, 0)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    plt.show()

    return means_magn, means_phase, frequencies

def impedance_vs_hours_by_complex(imp_magn, imp_phase, frequencies_studied, mean_window, complex_study):
    print("--------------------------------------------------------------------")
    print("------------- starting following impedance vs hours ----------------")
    print("--------------------------------------------------------------------")

    # Duration between each point in seconds
    duration_between_points = 590  # 9 minutes and 50 seconds

    # Initialization of time
    current_time = timedelta(seconds=0)

    # List of colors for each frequency
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'black', 'pink']

    # Creation of the time list only once
    time = []

    impedance_to_return = []
    for j in range(len(imp_magn[0])):
        time.append(current_time.total_seconds() / 3600)  # Convert time to hours
        current_time += timedelta(seconds=duration_between_points)

    if complex_study == 'yes':
        tab_complex = tab_to_complex(imp_magn, imp_phase)
        print("Impedances converted to complex successfully")
    else:
        tab_complex = imp_magn

    print(f"Analysis of {len(tab_complex[0])} points at {frequencies_studied} Hz")
    for i, imp_complex in enumerate(tab_complex):
        moving_average = []

        ############################ type of filtering to apply ###########################################

        ### MOVING AVERAGE -------------------------------------------------------------------

        moving_average_complex = np.convolve(imp_complex, np.ones(mean_window) / mean_window, mode='same')

        if mean_window == 1:
            print("No moving average")
        else:
            print(f"Moving average at {frequencies_studied[i]} Hz with window of {mean_window} pts: ok")

        ### convert back to magnitude:
        for j in range(len(moving_average_complex)):
            magn_avg, phase = retrieve_phase_magnitude(moving_average_complex[j])
            moving_average.append(round(magn_avg, 2))

        ### filter low points -------------------------------------------------------------------

        # max_threshold = 30
        # window_size = 3
        # magnitudes = filter_low_values(moving_average, max_threshold, window_size)

        ###### to filter low points (next 4 lines)
        # max_threshold = 20
        # window_size = 10
        # filtered_magnitudes = filter_low_values(moving_average, max_threshold, window_size)
        # plt.plot(time[:len(filtered_magnitudes)], filtered_magnitudes, marker='o', markersize=1, linestyle='-', label=str(freq_khz) + ' kHz')

        # Display the line with a different color for each frequency
        freq = round(frequencies_studied[i])
        freq_khz = round(freq / 1000, 2)
        impedance_to_return.append(moving_average)

    return impedance_to_return, time

def plot_difference(array1, array2):
    plt.figure(figsize=(10, 6))
    # Check if the two arrays have the same length
    print('array sizes:', len(array1))
    if len(array1) != len(array2):
        raise ValueError("The two arrays must have the same length.")

    # Calculate the difference between corresponding values
    differences = np.subtract(array1, array2)

    # Create an array of indices for the number of values
    indices = np.arange(len(array1))

    # Plot the graph of differences against the number of values
    plt.plot(indices, differences, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of values')
    plt.ylabel('Difference')
    plt.title('Difference between the values of the two arrays')

    # Display the graph in a separate window
    plt.show()

#################################################################################################   
#                                           recup debitmetre                                    #
#################################################################################################
 
def recup_vitesse_CO2(file):
    print(f"Attempting to read file: {file}")
    
    # Get the absolute path of the script directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Concatenate the text file name with the script directory path
    file_path = os.path.join(script_directory, file)
    
    print(f"Full file path: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return []

    # Initialize a list to store each element of the fourth column of each line in the file
    file_data = []
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line_num, line in enumerate(file, 1):
            # Split the line into words (or elements separated by a space)
            words = line.split()
            # If the line has at least 4 words, add the 4th word to the data list
            if len(words) >= 4:
                file_data.append(words[3].strip().lower())
            else:
                print(f"Warning: Line {line_num} does not have at least 4 elements: {line.strip()}")

    print(f"Number of data points read: {len(file_data)}")
    if file_data:
        print(f"First few data points: {file_data[:5]}")
    else:
        print("No data points read from the file.")

    # Remove the strings 'vitesse' and 'phase' if they exist
    file_data = [value for value in file_data if value not in ['vitesse', 'phase']]
    
    # Convert the remaining elements to floating-point numbers
    try:
        file_data = [float(value) for value in file_data]
    except ValueError as e:
        print(f"Error converting to float: {e}")
        print(f"Problematic values: {[value for value in file_data if not value.replace('.', '').isdigit()]}")
        return []
    
    return file_data

def plot_vitesse_CO2(file):
    # Retrieve the data
    data = recup_vitesse_CO2(file)

    # Generate times based on the number of data points (20 minutes between each point)
    times = [i * 20 / 60 for i in range(len(data))]

    # Create a figure with a shared axis on the right
    fig, ax1 = plt.subplots()

    # Plot the data on the left y-axis
    ax1.plot(times, data, label='CO2 Velocity', color='red')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('CO2 Velocity', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    plt.title('CO2 Release Velocity as a Function of Time')
    plt.show()
def plot_impedance_and_velocity(file, imp_magn, imp_phase, studied_frequencies, averaging_window, complex_study):
    # Retrieve CO2 velocity data
    CO2_velocity_data = recup_vitesse_CO2(file)

    # Retrieve impedance data
    impedance_data, time = impedance_vs_hours_by_complex(imp_magn, imp_phase, studied_frequencies, averaging_window, complex_study)
    
    print("data:", impedance_data)
    print("time:", time)

    # Create a figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CO2 velocity data if available
    if CO2_velocity_data and len(CO2_velocity_data) > 0:
        CO2_velocity_time = [i * 20 / 60 for i in range(len(CO2_velocity_data))]
        ax1.plot(CO2_velocity_time, CO2_velocity_data, label='CO2 Velocity', color='red')
        ax1.set_ylabel('CO2 Velocity', color='red')
        ax1.set_ylim(0, max(CO2_velocity_data) + 1)
        ax1.tick_params(axis='y', labelcolor='red')
    else:
        print("No CO2 velocity data available.")

    # Plot impedance data if available
    if impedance_data and len(impedance_data) > 0 and len(time) > 0:
        ax2 = ax1.twinx()
        ax2.scatter(time, impedance_data[0], label='Impedance', color='blue', s=5)
        ax2.set_ylabel('Impedance (ohms)', color='blue')
        ax2.set_ylim(min(impedance_data[0]) - 50, max(impedance_data[0]) + 50)
        ax2.tick_params(axis='y', labelcolor='blue')
    else:
        print("No impedance data available.")

    # Set common x-axis properties
    ax1.set_xlabel('Time (hours)')
    ax1.grid(True)

    # Set x-axis limits
    if CO2_velocity_data and len(CO2_velocity_data) > 0:
        ax1.set_xlim(min(CO2_velocity_time), max(CO2_velocity_time))
    elif time and len(time) > 0:
        ax1.set_xlim(min(time), max(time))

    # Add a combined legend for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if 'ax2' in locals() else ([], [])
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('CO2 Velocity and Impedance as a Function of Time')
    plt.tight_layout()
    plt.show()
    
    return time, impedance_data[0] if impedance_data and len(impedance_data) > 0 else []


#################################################################################################   
#                                           under development                                   #
#################################################################################################

def write_to_csv(values, csv_filename):
    """
    Write values to a CSV file.

    Parameters:
    - values (list or array): Array of values to be written to the CSV file.
    - csv_filename (str): Name of the CSV file.

    Returns:
    - None
    """
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the values to the CSV file
            csv_writer.writerow(values)

        print(f"Values successfully written to {csv_filename}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        
#################################################################################################   
#                                         Timothy Works                                #
#################################################################################################
import warnings
warnings.filterwarnings("ignore", message="Simulating circuit based on initial parameters")
# Function to calculate complex impedances from file data
def calculate_complex_impedances(file):
    # Get mean magnitudes, phases, and frequencies from the file
    mean_magnitudes, mean_phases, frequencies = mean_complex_file(file)
    
    # Calculate complex impedances using magnitude and phase
    complex_impedances = [impedance_complex(mag, phase) for mag, phase in zip(mean_magnitudes, mean_phases)]
    
    return complex_impedances, frequencies


# Function to plot Nyquist diagram for a given file
from impedance.visualization import plot_nyquist

def plot_nyquists(file):
    # Calculate complex impedances and get frequencies
    complex_impedances, frequencies, circuit = calculate_complex_impedances(file)
    
    # Use impedance.py's plot_nyquist function to create the plot
    plot_nyquist(frequencies, complex_impedances, ax=None, fit=circuit.predict(frequencies))
    plt.show()
    
# Function to plot multiple Nyquist plots for comparison
def plot_multi_nyquist(files,phase_shift= True):
    plt.figure(figsize=(12, 10))
    
    for file in files:
        # Load raw data from each file
        frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
        
        # Convert to numpy arrays for processing
        impedance_data = np.array(impedance_data)
        phase_data = np.array(phase_data)
        if phase_shift == True:
            phase_data = phase_data - 180  # Shift phase by -180 degrees
        
        # Calculate Z_real and Z_imag for all data points
        Z_real = impedance_data * np.cos(np.radians(phase_data))
        Z_imag = impedance_data * np.sin(np.radians(phase_data))
        
        # Plot data for this file with a single color
        label = os.path.basename(file)
        plt.plot(Z_real.flatten(), Z_imag.flatten(), 'o', label=label, markersize=3, alpha=0.7)

    # Set plot labels and title
    plt.xlabel('Real Part of Impedance (Ω)', fontsize=12)
    plt.ylabel('-Imaginary Part of Impedance (Ω)', fontsize=12)
    plt.title('Nyquist Plot', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Set aspect ratio to equal for proper Nyquist plot shape
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Print ranges for each file
    for file in files:
        frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
        impedance_data = np.array(impedance_data)
        phase_data = np.array(phase_data)
        Z_real = impedance_data * np.cos(np.radians(phase_data))
        Z_imag = -impedance_data * np.sin(np.radians(phase_data))
        
        print(f"File: {file}")
        print(f"Real Z range: {np.min(Z_real)} to {np.max(Z_real)} Ω")
        print(f"Imaginary Z range: {np.min(Z_imag)} to {np.max(Z_imag)} Ω")
        print("--------------------")
    
    plt.show()





# Function to generate distinct colors for plotting
from itertools import cycle

def get_distinct_colors(n):
    colors = []
    
    # Use a mix of colormaps for more variety
    cmaps = [plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Dark2, plt.cm.tab10]
    
    for cmap in cycle(cmaps):
        colors.extend(cmap(np.linspace(0, 1, 8)))
        if len(colors) >= n:
            break
    
    return colors[:n]

# Function to calculate and plot Bode plots for multiple files
def calculate_and_plot_bode_multi(files, phase_shift=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Get distinct colors for each file
    colors = get_distinct_colors(len(files))
    
    for file, color in zip(files, colors):
        # Load raw data from file
        frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
        
        # Convert to numpy arrays
        frequencies = np.array(frequencies)
        impedance_data = np.array(impedance_data)
        phase_data = np.array(phase_data)
        if phase_shift == True:
            phase_data = phase_data - 180  # Shift phase by -180 degrees
        
        # Ensure all arrays have the same length
        min_length = min(len(frequencies), impedance_data.shape[1], phase_data.shape[1])
        frequencies = frequencies[:min_length]
        impedance_data = impedance_data[:, :min_length]
        phase_data = phase_data[:, :min_length]
        
        # Calculate mean impedance and phase across all time points
        mean_impedance = np.mean(impedance_data, axis=0)
        mean_phase = np.mean(phase_data, axis=0)
        
        # Calculate cutoff frequency (assuming it's the frequency at -3dB point)
        max_impedance = np.max(mean_impedance)
        cutoff_value = max_impedance / np.sqrt(2)
        cutoff_indices = np.where(mean_impedance <= cutoff_value)[0]
        
        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            cutoff_frequency = frequencies[cutoff_index]
        else:
            print(f"Warning: Could not find cutoff frequency for {file}")
            cutoff_frequency = None
        
        # Plot mean data for each file with a single color
        label = os.path.basename(file)
        ax1.semilogx(frequencies, mean_impedance, label=label, color=color)
        ax2.semilogx(frequencies, mean_phase, label=label, color=color)
        
        # Add cutoff frequency line
        if cutoff_frequency is not None:
            ax1.axvline(x=cutoff_frequency, color=color, linestyle='--', alpha=0.7)
            ax2.axvline(x=cutoff_frequency, color=color, linestyle='--', alpha=0.7)
        
        # Print ranges and cutoff frequency for each file
        print(f"File: {file}")
        print(f"Frequency range: {min(frequencies)} to {max(frequencies)} Hz")
        print(f"Impedance range: {np.min(impedance_data)} to {np.max(impedance_data)} ohms")
        print(f"Phase range: {np.min(phase_data)} to {np.max(phase_data)} degrees")
        if cutoff_frequency is not None:
            print(f"Cutoff frequency: {cutoff_frequency:.2f} Hz")
        print(f"Max impedance: {max_impedance:.2f} ohms")
        print(f"Cutoff value: {cutoff_value:.2f} ohms")
        print("--------------------")

    # Set titles and labels for the plots
    ax1.set_title('Bode Plot of Impedance', fontsize=14)
    ax1.set_ylabel('|Z| (Ω)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Phase (degrees)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

        

# Function to plot comprehensive impedance analysis
def plot_impedance_analysis(file, time_point=None, start_time=None, end_time=None, num_plots=10, phase_shift=True):
    # Load fermentation data from file
    frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
    
    # Get the file name without the path
    file_name = os.path.basename(file)
    
    # Convert to numpy arrays for easier manipulation
    frequencies = np.array(frequencies)
    time_points = np.array(time_points)
    impedance_data = np.array(impedance_data)
    phase_data = np.array(phase_data)
    if phase_shift == True:
        phase_data = phase_data - 180  # Shift phase by -180 degrees
    
    # Trim frequencies to match impedance and phase data
    frequencies = frequencies[:impedance_data.shape[1]]

    # Convert time_points to hours if they're not already
    if np.max(time_points) > 24:  # Assuming times greater than 24 are in minutes
        time_points = time_points / 60  # Convert to hours

    if time_point is not None:
        # Find the closest time point to the specified time
        time_index = np.argmin(np.abs(time_points - time_point))
        selected_times = [time_points[time_index]]
        selected_impedance = [impedance_data[time_index]]
        selected_phase = [phase_data[time_index]]
    else:
        # If start_time and end_time are not provided, use the full range
        if start_time is None:
            start_time = np.min(time_points)
        if end_time is None:
            end_time = np.max(time_points)
        
        # Find indices within the specified time range
        time_mask = (time_points >= start_time) & (time_points <= end_time)
        selected_times = time_points[time_mask]
        selected_impedance = impedance_data[time_mask]
        selected_phase = phase_data[time_mask]
        
        # Select evenly spaced time points within the range
        if len(selected_times) > num_plots:
            indices = np.linspace(0, len(selected_times) - 1, num_plots, dtype=int)
            selected_times = selected_times[indices]
            selected_impedance = selected_impedance[indices]
            selected_phase = selected_phase[indices]

    print(f"Frequency range: {min(frequencies)} to {max(frequencies)} Hz")
    print(f"Impedance range: {np.min(impedance_data)} to {np.max(impedance_data)} ohms")
    print(f"Phase range: {np.min(phase_data)} to {np.max(phase_data)} degrees")

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Color map for time progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_times)))
    
    # Calculate cutoff frequency (assuming it's the frequency at -3dB point)
    cutoff_frequencies = []
    for impedance in selected_impedance:
        cutoff_index = np.argmin(np.abs(impedance - (np.max(impedance) / np.sqrt(2))))
        cutoff_frequencies.append(frequencies[cutoff_index])
    
    # Plot impedance magnitude
    for i, (time, color) in enumerate(zip(selected_times, colors)):
        ax1.semilogx(frequencies, selected_impedance[i], color=color, label=f't = {time:.2f} h')
        ax1.axvline(x=cutoff_frequencies[i], color='red', linestyle='--', alpha=0.5)
    
    ax1.set_title(f'Impedance Magnitude vs Frequency - {file_name}')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Impedance (Ω)')
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Plot phase
    for i, (time, color) in enumerate(zip(selected_times, colors)):
        ax2.semilogx(frequencies, selected_phase[i], color=color, label=f't = {time:.2f} h')
        ax2.axvline(x=cutoff_frequencies[i], color='red', linestyle='--', alpha=0.5)
    
    ax2.set_title(f'Impedance Phase vs Frequency - {file_name}')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Nyquist plot
    for i, (time, color) in enumerate(zip(selected_times, colors)):
        Z_real = selected_impedance[i] * np.cos(np.radians(selected_phase[i]))
        Z_imag = selected_impedance[i] * np.sin(np.radians(selected_phase[i]))
        ax3.plot(Z_real, Z_imag, color=color, label=f't = {time:.2f} h')
        ax3.scatter(Z_real[0], Z_imag[0], color=color, marker='o')
        ax3.scatter(Z_real[-1], Z_imag[-1], color=color, marker='s')

    ax3.set_title(f'Nyquist Plot - {file_name}')
    ax3.set_xlabel('Real Z (Ω)')
    ax3.set_ylabel('-Imaginary Z (Ω)')
    ax3.grid(True)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# For a specific time point:
# plot_impedance_analysis('your_file.csv', time_point=15)

# For a time range:
# plot_impedance_analysis('your_file.csv', start_time=10, end_time=20, num_plots=5)

# Function to fit circuit model to impedance data
from impedance import preprocessing
from impedance.models.circuits import CustomCircuit

def calculate_complex_impedances(file):
    mean_magnitudes, mean_phases, frequencies = mean_complex_file(file)
    
    # Calculate complex impedances
    complex_impedances = [impedance_complex(mag, phase) for mag, phase in zip(mean_magnitudes, mean_phases)]
    
    return complex_impedances, frequencies

def fit_circuit(frequencies, complex_impedances, order=1):
    # Prepare data for fitting
    frequencies = np.array(frequencies)
    Z = np.array(complex_impedances)
    
    # Define the circuit model based on the specified order
    if order == 1:
        circuit_string = 'R0-p(R1,CPE1)'
        initial_guess = [10, 100, 1e-5, 0.8]
    elif order == 2:
        circuit_string = 'R0-p(R1,CPE1)-p(R2,CPE2)'
        initial_guess = [10, 100, 1e-5, 0.8, 100, 1e-5, 0.8]
    elif order == 3:
        circuit_string = 'R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)'
        initial_guess = [10, 100, 1e-5, 0.8, 100, 1e-5, 0.8, 100, 1e-5, 0.8]
    else:
        raise ValueError("Order must be 1, 2, or 3")

    # Define the circuit model
    circuit = CustomCircuit(circuit_string, initial_guess=initial_guess)
    
    # Fit the circuit model to the data
    circuit.fit(frequencies, Z)
    
    # Get the fitted parameters
    fitted_params = circuit.parameters_
    
    return circuit, fitted_params



# Function to optimize circuit fitting using differential evolution
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter
import multiprocessing
import pandas as pd

def objective(params, frequencies, Z, circuit_string, weights):
    # Define objective function for optimization
    circuit = CustomCircuit(circuit_string, initial_guess=params)
    Z_fit = circuit.predict(frequencies)
    residuals = (np.abs(Z) - np.abs(Z_fit)) * weights
    return np.sum(residuals**2)

# Function to plot and fit impedance data
def plot_and_fit_impedance(file, order=2, time_point=None, phase_shift=True, smooth=False, window_length=15, polyorder=3, optimize=True):
    # Load fermentation data
    frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
    
    # Convert to numpy arrays
    frequencies = np.array(frequencies)
    impedance_data = np.array(impedance_data)
    phase_data = np.array(phase_data)
    
    if phase_shift:
        phase_data = phase_data - 180  # Shift phase by -180 degrees
    
    # If a specific time point is not provided, use the first one
    if time_point is None:
        time_index = 0
    else:
        time_index = np.argmin(np.abs(np.array(time_points) - time_point))
    
    # Ensure all arrays have the same length
    min_length = min(len(frequencies), impedance_data.shape[1], phase_data.shape[1])
    frequencies = frequencies[:min_length]
    impedance_data = impedance_data[:, :min_length]
    phase_data = phase_data[:, :min_length]
    
    # Calculate complex impedances
    Z = impedance_data[time_index] * np.exp(1j * np.radians(phase_data[time_index]))
    
    # Apply smoothing if requested
    if smooth:
        Z_real_smooth = savgol_filter(Z.real, window_length, polyorder)
        Z_imag_smooth = savgol_filter(Z.imag, window_length, polyorder)
        Z_smooth = Z_real_smooth + 1j * Z_imag_smooth
    else:
        Z_smooth = Z
    
    # Debug prints
    print(f"\nProcessing file: {file}")
    print(f"Frequency range: {np.min(frequencies):.2e} to {np.max(frequencies):.2e} Hz")
    print(f"Impedance magnitude range: {np.min(np.abs(Z_smooth)):.2f} to {np.max(np.abs(Z_smooth)):.2f} Ω")
    print(f"Phase range: {np.min(np.angle(Z_smooth, deg=True)):.2f} to {np.max(np.angle(Z_smooth, deg=True)):.2f} degrees")
    print(f"Length of frequencies: {len(frequencies)}")
    print(f"Length of impedance data: {len(Z_smooth)}")
    
    # Refined initial guesses
    R_total = np.max(Z_smooth.real)
    R0_guess = np.min(Z_smooth.real)
    R1_guess = (R_total - R0_guess) * 0.7
    R2_guess = (R_total - R0_guess) * 0.3
    CPE1_guess = 1 / (2 * np.pi * frequencies[np.argmax(-Z_smooth.imag)] * R1_guess)
    CPE2_guess = 1 / (2 * np.pi * frequencies[-1] * R2_guess)
    Wo_guess = np.sqrt(2) * R2_guess / np.sqrt(2 * np.pi * frequencies[-1])

    # Define the circuit model based on the specified order
    if order == 0:
        circuit_string = 'p(R1,C1)'
        initial_guess = [R_total, CPE1_guess]
        param_bounds = [(R_total * 0.1, R_total * 10), (CPE1_guess * 0.1, CPE1_guess * 10)]
    elif order == 1:
        circuit_string = 'R0-p(R1,CPE1)'
        initial_guess = [R0_guess, R1_guess, CPE1_guess, 0.8]
        param_bounds = [(0, R_total), (0, R_total), (1e-12, 1e-3), (0.5, 1)]
    elif order == 2:
        circuit_string = 'R0-p(R1,CPE1)-p(R2-Wo1,CPE2)'
        initial_guess = [R0_guess, R1_guess, CPE1_guess, 0.8, R2_guess, Wo_guess, 0.5, CPE2_guess, 0.8]
        param_bounds = [
            (0, R_total), (0, R_total), (1e-12, 1e-3), (0.5, 1),
            (0, R_total), (1e-6, 1e3), (0.1, 1),
            (1e-12, 1e-3), (0.5, 1)
        ]
    elif order == 3:
        circuit_string = 'R0-p(R1,CPE1)-p(R2,CPE2)-p(R3-Wo1,CPE3)'
        initial_guess = [R0_guess, R1_guess, CPE1_guess, 0.8, R2_guess, CPE2_guess, 0.8, R2_guess, Wo_guess, 0.5, CPE2_guess, 0.8]
        param_bounds = [
            (0, R_total), (0, R_total), (1e-12, 1e-3), (0.5, 1),
            (0, R_total), (1e-12, 1e-3), (0.5, 1),
            (0, R_total), (1e-6, 1e3), (0.1, 1),
            (1e-12, 1e-3), (0.5, 1)
        ]
    else:
        raise ValueError("Order must be 0, 1, 2, or 3")

    # Precompute weights
    weights = 1 / np.abs(Z_smooth)

    if optimize:
        # Determine the number of CPU cores to use
        num_cores = multiprocessing.cpu_count()
        
        # Check if param_bounds are valid
        for i, (low, high) in enumerate(param_bounds):
            if low >= high:
                print(f"Warning: Invalid bounds for parameter {i}: [{low}, {high}]")
                param_bounds[i] = (low, low * 10)  # Adjust the upper bound
        
        try:
            # Use differential evolution with refined parameters
            result = differential_evolution(
                objective,
                param_bounds, 
                args=(frequencies, Z_smooth, circuit_string, weights),
                maxiter=500,
                popsize=10,
                tol=1e-4, 
                mutation=(0.5, 1.2), 
                recombination=0.7,
                updating='deferred',
                workers=num_cores  # Use all available CPU cores
            )
        
            # Create and fit the circuit with the optimal parameters
            circuit = CustomCircuit(circuit_string, initial_guess=result.x)
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            print("Falling back to initial guess")
            circuit = CustomCircuit(circuit_string, initial_guess=initial_guess)
    else:
        # Create and fit the circuit with the initial guess
        circuit = CustomCircuit(circuit_string, initial_guess=initial_guess)
        
    circuit.fit(frequencies, Z_smooth)
    
    # Generate fitted impedance data
    Z_fit = circuit.predict(frequencies)
    
    # Calculate fitting error
    error = np.mean(np.abs(Z_smooth - Z_fit) / np.abs(Z_smooth)) * 100
    
    # Create a dictionary to store the data
    data = {
        "File": file,
        "Order": order,
        "Time Point": time_point,
        "Phase Shift": phase_shift,
        "Smooth": smooth,
        "Window Length": window_length,
        "Polyorder": polyorder,
        "Optimize": optimize,
        "Fitting Error": f"{error:.2f}%"
    }
    
    # Plot Nyquist
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_nyquist(Z, fmt='o', scale=1, ax=ax, label='Raw Data')
    if smooth:
        plot_nyquist(Z_smooth, fmt='s', scale=1, ax=ax, label='Smoothed Data')
    plot_nyquist(Z_fit, fmt='-', scale=1, ax=ax, label=f'Fit (Error: {error:.2f}%)')
    ax.scatter(Z.real[0], -Z.imag[0], color='blue', marker='o', s=100, label='Start')
    ax.scatter(Z.real[-1], -Z.imag[-1], color='blue', marker='s', s=100, label='End')
    ax.set_xlabel('Real Z (Ω)')
    ax.set_ylabel('-Imaginary Z (Ω)')
    ax.legend()
    ax.set_title(f'Nyquist Plot with Order {order} Circuit Fit\n{file}')
    ax.grid(True)
    ax.axis('equal')

    print(f"\nFitting Error: {error:.2f}%")
    print("")
    print(circuit)

    plt.tight_layout()
    plt.show()

    return circuit, data, frequencies  # Return frequencies as well

# Function to get circuit parameters based on order
def get_circuit_parameters(order):
    if order == 0:
        return ['R1', 'C1']
    elif order == 1:
        return ['R0', 'R1', 'CPE1-T', 'CPE1-P']
    elif order == 2:
        return ['R0', 'R1', 'CPE1-T', 'CPE1-P', 'R2', 'Wo1-R', 'Wo1-T', 'CPE2-T', 'CPE2-P']
    elif order == 3:
        return ['R0', 'R1', 'CPE1-T', 'CPE1-P', 'R2', 'CPE2-T', 'CPE2-P', 'R3', 'Wo1-R', 'Wo1-T', 'CPE3-T', 'CPE3-P']
    else:
        raise ValueError("Order must be 0, 1, 2, or 3")

# Function to write analysis results to Excel
def write_to_excel(data_list, output_file="impedance_analysis_results.xlsx"):
    df = pd.DataFrame(data_list)
    
    # Identify all existing parameter columns
    param_columns = [col for col in df.columns if col.endswith('_initial') or col.endswith('_fitted')]
    param_names = sorted(set(col.split('_')[0] for col in param_columns))
    
    # Reorder columns
    column_order = ['File', 'Sugar', 'Nitrogen', 'Alcohol', 'Order', 'Fitting Error']
    for param in param_names:
        if f'{param}_initial' in df.columns:
            column_order.append(f'{param}_initial')
        if f'{param}_fitted' in df.columns:
            column_order.append(f'{param}_fitted')
    
    # Add any remaining columns
    remaining_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_columns)
    
    # Reorder the DataFrame columns, only including existing columns
    df = df.reindex(columns=[col for col in column_order if col in df.columns])
    
    df.to_excel(output_file, index=False)
    print(f"Data has been written to {output_file}")

from scipy.interpolate import interp1d

import re

def analyze_cutoff_frequency_evolution(files, parameter_name):
    cutoff_frequencies = []
    concentrations = []
    
    for file in files:
        # Extract concentration from filename
        match = re.search(r'(\d+)', file)
        if match:
            concentration = int(match.group(1))
            concentrations.append(concentration)
        else:
            print(f"Couldn't extract concentration from filename: {file}")
            continue
        
        # Load data
        data = get_file_data(file)
        
        frequencies = string_to_float(data[1])  # Assuming second row is frequencies
        
        # Extract impedance data
        Z_data = data[2][1:]  # Assuming third row is impedance magnitude data, skip first column
        Z_mag = string_to_float(Z_data)
        
        # Check if lengths match and trim if necessary
        if len(frequencies) != len(Z_mag):
            print(f"Warning: Frequency and impedance arrays have different lengths in file {file}")
            min_length = min(len(frequencies), len(Z_mag))
            frequencies = frequencies[:min_length]
            Z_mag = Z_mag[:min_length]
        
        # Find the maximum impedance
        Z_max = np.max(Z_mag)
        
        # Calculate the cutoff frequency (-3dB point)
        cutoff_value = Z_max / np.sqrt(2)
        
        # Find the cutoff frequency using simple search
        idx_cutoff = np.argmin(np.abs(Z_mag - cutoff_value))
        cutoff_freq = frequencies[idx_cutoff]
        
        cutoff_frequencies.append(cutoff_freq)
    
    # Sort concentrations and cutoff frequencies together
    concentrations, cutoff_frequencies = zip(*sorted(zip(concentrations, cutoff_frequencies)))
    
    # Plot the evolution of cutoff frequency
    plt.figure(figsize=(10, 6))
    plt.plot(concentrations, cutoff_frequencies, 'o-')
    plt.xlabel(f'{parameter_name} Concentration')
    plt.ylabel('Cutoff Frequency (Hz)')
    plt.title(f'Evolution of Cutoff Frequency vs {parameter_name} Concentration')
    plt.xscale('linear')  # or 'log' if needed
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # Print the results
    print(f"\nCutoff Frequencies for {parameter_name}:")
    for conc, freq in zip(concentrations, cutoff_frequencies):
        print(f"Concentration: {conc}, Cutoff Frequency: {freq:.2f} Hz")
    
    return concentrations, cutoff_frequencies




def plot_element_impact(files, element_name, concentrations):
    """
    Plot the impact of different concentrations of an element on impedance spectra.
    
    Args:
    files (list): List of file names containing impedance data for different concentrations.
    element_name (str): Name of the element being varied (e.g., 'Sugar', 'Alcohol', 'Nitrogen').
    concentrations (list): List of concentrations corresponding to each file.
    
    Returns:
    None (displays the plots)
    """
    
    # Set up the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    
    for file, concentration, color in zip(files, concentrations, colors):
        # Load data using load_fermentation_data
        frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
        
        # Convert to numpy arrays
        frequencies = np.array(frequencies)
        impedance_data = np.array(impedance_data)
        phase_data = np.array(phase_data)
        
        # Ensure all arrays have the same length
        min_length = min(len(frequencies), impedance_data.shape[1], phase_data.shape[1])
        frequencies = frequencies[:min_length]
        impedance_data = impedance_data[:, :min_length]
        phase_data = phase_data[:, :min_length]
        
        # Calculate average impedance and phase across all time points
        mean_impedance = np.mean(impedance_data, axis=0)
        mean_phase = np.mean(phase_data, axis=0)
        
        # Calculate complex impedances
        complex_impedances = mean_impedance * np.exp(1j * np.radians(mean_phase))
        
        # Bode plot - Magnitude
        ax1.loglog(frequencies, mean_impedance, color=color, label=f'{concentration}')
        
        # Bode plot - Phase
        ax2.semilogx(frequencies, mean_phase, color=color, label=f'{concentration}')
        
        # Nyquist plot
        ax3.plot(complex_impedances.real, -complex_impedances.imag, color=color, label=f'{concentration}')
        ax3.scatter(complex_impedances.real[0], -complex_impedances.imag[0], color=color, marker='o')
        ax3.scatter(complex_impedances.real[-1], -complex_impedances.imag[-1], color=color, marker='s')
        
        print(f"File: {file}")
        print(f"Number of frequency points: {len(frequencies)}")
        print(f"Number of impedance points: {len(mean_impedance)}")
        print(f"Frequency range: {frequencies[0]} to {frequencies[-1]} Hz")
        print(f"Impedance range: {np.min(mean_impedance)} to {np.max(mean_impedance)} ohms")
        print("--------------------")
    
    # Customize plots
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ω)')
    ax1.set_title(f'Bode Plot - Magnitude\nEffect of {element_name}')
    ax1.legend(title=f'{element_name} Concentration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title(f'Bode Plot - Phase\nEffect of {element_name}')
    ax2.legend(title=f'{element_name} Concentration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    ax3.set_xlabel('Real Z (Ω)')
    ax3.set_ylabel('-Imaginary Z (Ω)')
    ax3.set_title(f'Nyquist Plot\nEffect of {element_name}')
    ax3.legend(title=f'{element_name} Concentration', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
def load_fermentation_data(file):
    # Load raw data from the file
    file_data = get_file_data(file)
    
    # Extract frequency data from the second row and convert to float
    frequencies = string_to_float(file_data[1])
    
    # Initialize lists to store time points, impedance data, and phase data
    time_points = []
    impedance_data = []
    phase_data = []
    
    # Iterate through the file data, starting from the third row
    # Each measurement consists of 4 rows: time, impedance, phase, and an empty row
    for i in range(2, len(file_data), 4):
        # Extract time point from the first column
        time_points.append(float(file_data[i][0]))
        
        # Extract impedance data from the current row (excluding the first column)
        impedance_data.append(string_to_float(file_data[i][1:]))
        
        # Extract phase data from the next row (excluding the first column)
        phase_data.append(string_to_float(file_data[i+1][1:]))
    
    # Return the extracted data
    return frequencies, time_points, impedance_data, phase_data

def load_and_process_data(file, phase_shift=True):
    frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
    
    # Convert to numpy arrays
    frequencies = np.array(frequencies)
    impedance_data = np.array(impedance_data)
    phase_data = np.array(phase_data)
    
    if phase_shift:
        phase_data = phase_data - 180  # Shift phase by -180 degrees
    
    # Ensure all arrays have the same length
    min_length = min(len(frequencies), impedance_data.shape[1], phase_data.shape[1])
    frequencies = frequencies[:min_length]
    impedance_data = impedance_data[:, :min_length]
    phase_data = phase_data[:, :min_length]
    
    # Calculate average impedance and phase across all time points
    mean_impedance = np.mean(impedance_data, axis=0)
    mean_phase = np.mean(phase_data, axis=0)
    
    # Calculate complex impedances
    Z_real = mean_impedance * np.cos(np.radians(mean_phase))
    Z_imag = mean_impedance * np.sin(np.radians(mean_phase))
    complex_impedances = Z_real + 1j * Z_imag
    
    return frequencies, mean_impedance, mean_phase, complex_impedances

def plot_impedance_spectra(ax, frequencies, complex_impedances, label, color):
    # Plot the magnitude of impedance on a log-log scale
    ax[0].loglog(frequencies, np.abs(complex_impedances), color=color, label=label)
    
    # Plot the phase angle of impedance on a semilog scale
    ax[1].semilogx(frequencies, np.angle(complex_impedances, deg=True), color=color, label=label)
    
    # Plot the Nyquist plot (real vs. negative imaginary part of impedance)
    ax[2].plot(complex_impedances.real, -complex_impedances.imag, color=color, label=label)
    
    # Mark the starting point (lowest frequency) on the Nyquist plot with a circle
    ax[2].scatter(complex_impedances.real[0], -complex_impedances.imag[0], color=color, marker='o')
    
    # Mark the ending point (highest frequency) on the Nyquist plot with a square
    ax[2].scatter(complex_impedances.real[-1], -complex_impedances.imag[-1], color=color, marker='s')


        

def analyze_single_elements(files, concentrations, element_name):
    # Create a new figure with specified size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each file and corresponding concentration
    for file, conc in zip(files, concentrations):
        # Load and process data from the file
        frequencies, mean_impedance, _, _ = load_and_process_data(file)
        
        # Plot the impedance magnitude vs frequency on a semilog scale
        ax.semilogx(frequencies, mean_impedance, label=f'{conc} g/L')

    # Set labels and title for the plot
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Z| (Ω)')
    ax.set_title(f'Impedance Magnitude vs Frequency - {element_name}')
    
    # Add a legend to the plot
    ax.legend()
    
    # Add a grid to the plot for better readability
    ax.grid(True)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    
    # Display the plot
    plt.show()

    # Perform additional analysis on impedance variation
    analyze_impedance_variation(files, concentrations, element_name, frequencies)

def analyze_impedance_variation(files, concentrations, element_name, frequencies):
    # List to store mean impedance data for each concentration
    impedances = []
    for file in files:
        # Load and process data from each file
        _, mean_impedance, _, _ = load_and_process_data(file)
        impedances.append(mean_impedance)

    print(f"\nAnalysis for {element_name}:")
    
    # Determine if impedance generally increases or decreases with concentration
    if np.mean(impedances[-1]) > np.mean(impedances[0]):
        print(f"Increasing {element_name} concentration generally increases impedance.")
    else:
        print(f"Increasing {element_name} concentration generally decreases impedance.")

    # Compare impedance between consecutive concentrations
    for i in range(1, len(concentrations)):
        low_conc = concentrations[i-1]
        high_conc = concentrations[i]
        
        print(f"\nComparing {low_conc} g/L to {high_conc} g/L:")
        
        # Analyze two frequency ranges: 1-40 kHz and 40 kHz to max frequency
        freq_ranges = [(1000, 40000), (40000, max(frequencies))]
        for start_freq, end_freq in freq_ranges:
            # Find indices corresponding to start and end frequencies
            start_idx = np.argmin(np.abs(np.array(frequencies) - start_freq))
            end_idx = np.argmin(np.abs(np.array(frequencies) - end_freq))
            
            # Calculate ratio of mean impedances in the frequency range
            ratio = np.mean(impedances[i][start_idx:end_idx]) / np.mean(impedances[i-1][start_idx:end_idx])
            
            print(f"From {start_freq/1000:.1f}kHz to {end_freq/1000:.1f}kHz:")
            print(f"  The magnitude is {'increased' if ratio > 1 else 'decreased'} by a factor of {ratio:.2f}")

        # Find and report the highest magnitude value and its frequency
        max_idx = np.argmax(impedances[i])
        max_freq = frequencies[max_idx]
        max_imp = impedances[i][max_idx]
        print(f"The highest magnitude value is {max_imp:.2f}Ω at {max_freq/1000:.2f}kHz")

        # Calculate and report the cutoff frequency (-3dB point)
        cutoff_value = max_imp / np.sqrt(2)
        cutoff_idx = np.argmin(np.abs(impedances[i] - cutoff_value))
        cutoff_freq = frequencies[cutoff_idx]
        print(f"The cutoff frequency is approximately {cutoff_freq/1000:.2f}kHz")

def analyze_combinations(files, conditions):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    for file, condition in zip(files, conditions):
        # Load and process data for each file
        frequencies, mean_impedance, _, _ = load_and_process_data(file)
        label = f"S:{condition['S']}, N:{condition['N']}, A:{condition['A']}"
        
        # Plot all data on the first subplot
        ax1.semilogx(frequencies, mean_impedance, label=label)
        
        # Plot data on the second subplot, excluding the high-impedance condition
        if condition['N'] != 0 or (condition['S'] == 0 and condition['A'] == 0):
            ax2.semilogx(frequencies, mean_impedance, label=label)

    # Set labels and title for the first subplot
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ω)')
    ax1.set_title('Impedance Magnitude vs Frequency - All Combinations')
    ax1.legend()
    ax1.grid(True)

    # Set labels and title for the second subplot
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('|Z| (Ω)')
    ax2.set_title('Impedance Magnitude vs Frequency - Excluding High Impedance Condition')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Additional analysis for the high-impedance condition
    high_imp_condition = next((cond for cond in conditions if cond['N'] == 0 and cond['S'] != 0), None)
    if high_imp_condition:
        high_imp_file = files[conditions.index(high_imp_condition)]
        frequencies, mean_impedance, _, _ = load_and_process_data(high_imp_file)
        
        # Plot high-impedance condition separately
        plt.figure(figsize=(10, 6))
        plt.semilogx(frequencies, mean_impedance)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('|Z| (Ω)')
        plt.title(f"High Impedance Condition: S:{high_imp_condition['S']}, N:{high_imp_condition['N']}, A:{high_imp_condition['A']}")
        plt.grid(True)
        plt.show()

        # Print analysis for high impedance condition
        print(f"\nAnalysis for high impedance condition (S:{high_imp_condition['S']}, N:{high_imp_condition['N']}, A:{high_imp_condition['A']}):")
        print(f"Maximum impedance: {max(mean_impedance):.2f} Ω")
        print(f"Minimum impedance: {min(mean_impedance):.2f} Ω")
        print(f"Impedance range: {max(mean_impedance) - min(mean_impedance):.2f} Ω")

def sensitivity_analysis():
    # Define base file and files for different elements
    base_file = 'S3c8.csv'
    sugar_files = ['S3c9.csv', 'S3c10.csv']
    nitrogen_files = ['S3c11.csv', 'S3c12.csv']
    alcohol_files = ['S3c13.csv', 'S3c14.csv']

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    # Load base data
    base_freq, base_imp, _, _ = load_and_process_data(base_file)
    
    # Plot sensitivity analysis for each element
    for ax, files, element in zip([ax1, ax2, ax3], [sugar_files, nitrogen_files, alcohol_files], ['Sugar', 'Nitrogen', 'Alcohol']):
        ax.semilogx(base_freq, base_imp, label='Base (S3c8)')
        for file in files:
            freq, imp, _, _ = load_and_process_data(file)
            ax.semilogx(freq, imp, label=file[:-4])
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|Z| (Ω)')
        ax.set_title(f'Sensitivity Analysis - {element}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate and print sensitivity analysis results
    print("\nSensitivity Analysis:")
    for element, files in zip(['Sugar', 'Nitrogen', 'Alcohol'], [sugar_files, nitrogen_files, alcohol_files]):
        print(f"\n{element} Sensitivity:")
        base_imp = load_and_process_data(base_file)[1]
        for file in files:
            imp = load_and_process_data(file)[1]
            low_freq_change = np.mean(imp[:40] / base_imp[:40])
            high_freq_change = np.mean(imp[40:] / base_imp[40:])
            print(f"  {file[:-4]}:")
            print(f"    Low frequency (1-40 kHz) change: {low_freq_change:.2f}x")
            print(f"    High frequency (40-178 kHz) change: {high_freq_change:.2f}x")

def fermentation_steps():
    # Define files and labels for two fermentation paths
    files_path1 = ['S3c15.csv', 'S3c17.csv', 'S3c18.csv']
    files_path2 = ['S3c16.csv', 'S3c17.csv', 'S3c18.csv']
    labels_path1 = ['Initial 1', 'Intermediate', 'Final']
    labels_path2 = ['Initial 2', 'Intermediate', 'Final']

    plt.figure(figsize=(12, 6))

    # Plot path 1
    for file, label in zip(files_path1, labels_path1):
        freq, imp, _, _ = load_and_process_data(file)
        plt.semilogx(freq, imp, label=f'Path 1: {label}', linestyle='-')

    # Plot path 2
    for file, label in zip(files_path2, labels_path2):
        freq, imp, _, _ = load_and_process_data(file)
        plt.semilogx(freq, imp, label=f'Path 2: {label}', linestyle='--')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (Ω)')
    plt.title('Impedance Changes During Fermentation Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nFermentation Steps Analysis:")
    
    # Analyze path 1
    print("\nPath 1 (S3c15 -> S3c17 -> S3c18):")
    base_imp = load_and_process_data(files_path1[0])[1]
    for file, label in zip(files_path1[1:], labels_path1[1:]):
        imp = load_and_process_data(file)[1]
        low_freq_change = np.mean(imp[:40] / base_imp[:40])
        high_freq_change = np.mean(imp[40:] / base_imp[40:])
        print(f"\n{label}:")
        print(f"  Low frequency (1-40 kHz) change: {low_freq_change:.2f}x")
        print(f"  High frequency (40-178 kHz) change: {high_freq_change:.2f}x")

    # Analyze path 2
    print("\nPath 2 (S3c16 -> S3c17 -> S3c18):")
    base_imp = load_and_process_data(files_path2[0])[1]
    for file, label in zip(files_path2[1:], labels_path2[1:]):
        imp = load_and_process_data(file)[1]
        low_freq_change = np.mean(imp[:40] / base_imp[:40])
        high_freq_change = np.mean(imp[40:] / base_imp[40:])
        print(f"\n{label}:")
        print(f"  Low frequency (1-40 kHz) change: {low_freq_change:.2f}x")
        print(f"  High frequency (40-178 kHz) change: {high_freq_change:.2f}x")


def calculate_equivalent_impedance(*impedances):
    """
    Calculate the equivalent impedance for parallel connection of two or more impedances.
    Each argument should be a numpy array of complex impedances.
    """
    if len(impedances) < 2:
        raise ValueError("At least two impedances are required for parallel calculation.")
    
    total_inverse = sum(1/Z for Z in impedances)
    return 1 / total_inverse
def get_complex_impedance(file, time_point=0):
    """
    Load impedance data from a file using load_fermentation_data() function
    and return complex impedance array for a specific time point.
    """
    frequencies, time_points, impedance_data, phase_data = load_fermentation_data(file)
    Z_mag = np.array(impedance_data[time_point])
    Z_phase = np.array(phase_data[time_point])
    
    # Ensure all arrays have the same length
    min_length = min(len(frequencies), len(Z_mag), len(Z_phase))
    frequencies = frequencies[:min_length]
    Z_mag = Z_mag[:min_length]
    Z_phase = Z_phase[:min_length]
    
    Z = Z_mag * np.exp(1j * np.radians(Z_phase))
    return frequencies, Z
from scipy import interpolate

def find_intersection(x, y1, y2):
    """Find the intersection point of two curves."""
    diff = y1 - y2
    # Find where the difference changes sign
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) > 0:
        # Get the index just before the sign change
        idx = sign_changes[0]
        # Interpolate to find a more precise x-value
        x_interp = np.linspace(x[idx], x[idx+1], 1000)
        f1 = interpolate.interp1d(x[idx:idx+2], y1[idx:idx+2])
        f2 = interpolate.interp1d(x[idx:idx+2], y2[idx:idx+2])
        y1_interp = f1(x_interp)
        y2_interp = f2(x_interp)
        idx_interp = np.argmin(np.abs(y1_interp - y2_interp))
        return x_interp[idx_interp]
    else:
        return None
    

def rc_parallel_impedance(R, C, f):
    """Calculate impedance of an RC parallel circuit."""
    omega = 2 * np.pi * f
    return R / (1 + 1j * omega * R * C)

def get_initial_guess(frequencies, Z):
    """Calculate initial guess for parallel RC circuit."""
    # Estimate R as the maximum magnitude of Z
    R_guess = np.max(np.abs(Z))
    # Find frequency at peak of imaginary part
    f_peak = frequencies[np.argmax(-Z.imag)]
    # Estimate C using the peak frequency and estimated R
    C_guess = 1 / (2 * np.pi * f_peak * R_guess)
    return R_guess, C_guess

def parallel_rc(R1, C1, R2, C2, f):
    """Calculate the equivalent impedance of two RC circuits in parallel"""
    # Calculate impedance of first RC circuit
    Z1 = R1 / (1 + 1j * 2 * np.pi * f * R1 * C1)
    # Calculate impedance of second RC circuit
    Z2 = R2 / (1 + 1j * 2 * np.pi * f * R2 * C2)
    # Return equivalent impedance of parallel combination
    return 1 / (1/Z1 + 1/Z2)

def compare_rc_circuits(circuit1, circuit2, circuit_combined, freq_range, labels):
    """
    Compare the impedance of two parallel RC circuits with a single RC circuit.
    
    Parameters:
    circuit1, circuit2: CustomCircuit objects for individual solutions
    circuit_combined: CustomCircuit object for the combined solution
    freq_range: Array of frequencies to calculate impedance over
    labels: Dictionary with labels for plotting
    """
    # Calculate impedances
    Z1 = circuit1.predict(freq_range)
    Z2 = circuit2.predict(freq_range)
    Z14 = circuit_combined.predict(freq_range)
    
    # Calculate parallel combination of Z1 and Z2
    Z_parallel = 1 / (1/Z1 + 1/Z2)
    
    # Calculate relative difference
    rel_diff = np.abs(Z14 - Z_parallel) / np.abs(Z14) * 100
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.loglog(freq_range, np.abs(Z_parallel), label=f'Parallel ({labels["file1"]} + {labels["file2"]})')
    plt.loglog(freq_range, np.abs(Z14), label=labels["file_combined"])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (Ω)')
    plt.title('Comparison of Impedance Magnitudes')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogx(freq_range, rel_diff)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Difference in Impedance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Average relative difference: {np.mean(rel_diff):.2f}%")
    print(f"Maximum relative difference: {np.max(rel_diff):.2f}%")
    print(f"Minimum relative difference: {np.min(rel_diff):.2f}%")

def fit_and_compare_rc_circuits(file1, file2, file_combined, labels):
    # Load data
    freq_1, Z_1 = get_complex_impedance(file1)
    freq_2, Z_2 = get_complex_impedance(file2)
    freq_combined, Z_combined = get_complex_impedance(file_combined)

    # Ensure all frequency arrays are the same length
    min_length = min(len(freq_1), len(freq_2), len(freq_combined))
    freq_1, Z_1 = freq_1[:min_length], Z_1[:min_length]
    freq_2, Z_2 = freq_2[:min_length], Z_2[:min_length]
    freq_combined, Z_combined = freq_combined[:min_length], Z_combined[:min_length]

    # Get initial guesses
    R1_guess, C1_guess = get_initial_guess(freq_1, Z_1)
    R2_guess, C2_guess = get_initial_guess(freq_2, Z_2)
    R_combined_guess, C_combined_guess = get_initial_guess(freq_combined, Z_combined)

    # Fit RC parallel circuit models
    circuit_string = 'p(R1,C1)'
    circuit1 = CustomCircuit(circuit_string, initial_guess=[R1_guess, C1_guess])
    circuit2 = CustomCircuit(circuit_string, initial_guess=[R2_guess, C2_guess])
    circuit_combined = CustomCircuit(circuit_string, initial_guess=[R_combined_guess, C_combined_guess])
    
    circuit1.fit(freq_1, Z_1)
    circuit2.fit(freq_2, Z_2)
    circuit_combined.fit(freq_combined, Z_combined)

    # Extract fitted parameters
    R1, C1 = circuit1.parameters_
    R2, C2 = circuit2.parameters_
    R_combined, C_combined = circuit_combined.parameters_

    # Print fitted parameters
    print(f"\nFitted parameters for {labels['file1']}:")
    print(f"R = {R1:.2e}, C = {C1:.2e}")
    print(f"\nFitted parameters for {labels['file2']}:")
    print(f"R = {R2:.2e}, C = {C2:.2e}")
    print(f"\nFitted parameters for {labels['file_combined']}:")
    print(f"R = {R_combined:.2e}, C = {C_combined:.2e}")

    # Generate frequency range for comparison
    freq_range = np.logspace(np.log10(min(freq_1[0], freq_2[0], freq_combined[0])),
                             np.log10(max(freq_1[-1], freq_2[-1], freq_combined[-1])),
                             1000)

    # Calculate impedance for parallel combination of circuit1 and circuit2
    Z1 = circuit1.predict(freq_range)
    Z2 = circuit2.predict(freq_range)
    Z_parallel = 1 / (1/Z1 + 1/Z2)

    # Calculate impedance for circuit_combined
    Z14 = circuit_combined.predict(freq_range)

    # Calculate relative difference
    rel_diff = np.abs(Z14 - Z_parallel) / np.abs(Z14) * 100

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.loglog(freq_range, np.abs(Z_parallel), label=f'Parallel ({labels["file1"]} + {labels["file2"]})')
    plt.loglog(freq_range, np.abs(Z14), label=labels['file_combined'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (Ω)')
    plt.title('Comparison of Impedance Magnitudes')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogx(freq_range, rel_diff)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Difference in Impedance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nComparison Statistics:")
    print(f"Average relative difference: {np.mean(rel_diff):.2f}%")
    print(f"Maximum relative difference: {np.max(rel_diff):.2f}%")
    print(f"Minimum relative difference: {np.min(rel_diff):.2f}%")
def compare_impedances(file1, file2, file_combined, labels):
    """
    Compare impedances of individual components with a combined solution.
    
    Parameters:
    file1, file2: Filenames for individual component data
    file_combined: Filename for combined solution data
    labels: Dictionary with labels for plotting
    """
    # Load data
    freq_1, Z_1 = get_complex_impedance(file1)
    freq_2, Z_2 = get_complex_impedance(file2)
    freq_combined, Z_combined = get_complex_impedance(file_combined)

    # Ensure all frequency arrays are the same length
    min_length = min(len(freq_1), len(freq_2), len(freq_combined))
    freq_1, Z_1 = freq_1[:min_length], Z_1[:min_length]
    freq_2, Z_2 = freq_2[:min_length], Z_2[:min_length]
    freq_combined, Z_combined = freq_combined[:min_length], Z_combined[:min_length]

    # Calculate equivalent impedance
    Z_eq = calculate_equivalent_impedance(Z_1, Z_2)

    # Calculate relative difference
    rel_diff = np.abs(Z_combined - Z_eq) / np.abs(Z_combined) * 100

    # Find intersection point
    intersection_freq = find_intersection(freq_combined, np.abs(Z_combined), np.abs(Z_eq))

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.semilogx(freq_combined, np.abs(Z_combined), label=labels['measured'])
    plt.semilogx(freq_combined, np.abs(Z_eq), label=labels['calculated'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|Z| (Ω)')
    plt.title('Comparison of Measured and Calculated Impedance')
    plt.legend()
    plt.grid(True)

    if intersection_freq is not None:
        intersection_z = np.interp(intersection_freq, freq_combined, np.abs(Z_combined))
        plt.plot(intersection_freq, intersection_z, 'ro', markersize=10)
        plt.annotate(f'{intersection_freq:.2f} Hz', 
                     (intersection_freq, intersection_z),
                     xytext=(5, 5), textcoords='offset points')
        print(f"Intersection frequency: {intersection_freq:.2f} Hz")
    else:
        print("No intersection found")

    plt.show()

    # Plot relative difference
    plt.figure(figsize=(12, 6))
    plt.semilogx(freq_combined, rel_diff)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Difference between Measured and Calculated Impedance')
    plt.grid(True)
    plt.show()

    print(f"Average relative difference: {np.mean(rel_diff):.2f}%")
    print(f"Maximum relative difference: {np.max(rel_diff):.2f}%")
    print(f"Minimum relative difference: {np.min(rel_diff):.2f}%")
    



#########################################################################################################################   
###--------------------------------------Timothy Work Under Development-----------------------------------------------###
############################################################################################### ######################### 




    


###############################################################################################      
###--------------------------------------MAIN-----------------------------------------------###
###############################################################################################   

def main():
    # plot_CO2_velocity('R1.txt')
    ####### monitoring measurements over time
    # time_monitoring('gazeux2H.csv', 1)  # set to 1 to display CO2 release on the same graph
    #time_monitoring('FERMENTATION.csv', 1)  # set to 1 to display CO2 release on the same graph

    ####### Averages of measurements in a folder: used for references or measurements on static solutions
    # averages('references before after fermentation')
    # averages('sucre_18janvier')
    # averages('alcool_15janvier')
    # averages('azote')
    # averages('test_1000')
    # averages('RC')
    # averages('test_1000')

    # csv_averages('81112')
    
    #### Study of the influence of concentration  modes: Admittance / Impedance
    #impedance_vs_concentration(mode='Admittance', freq_to_study=[1000])
    #impedance_vs_concentration(mode='Impedance', freq_to_study=[1000])   
    # csv_for_scilab('81314')
    #file_path='C:/Users/chiu-wing-timothy.ho/Downloads/Fermentation data and Bode analysis/IMPORTANT/mesures statiques 12 dec 2024/Sonde2/S2c2.csv'
    #base = ['RC.csv']
    """
 # Single element analysis
    sugar_files = ['S3demine.csv', 'S3c2.csv', 'S3c3.csv']
    sugar_concentrations = [0, 150, 300]
    analyze_single_elements(sugar_files, sugar_concentrations, 'Sugar')

    nitrogen_files = ['S3demine.csv', 'S3c4.csv', 'S3c5.csv']
    nitrogen_concentrations = [0, 0.25, 0.5]
    analyze_single_elements(nitrogen_files, nitrogen_concentrations, 'Nitrogen')

    alcohol_files = ['S3demine.csv', 'S3c6.csv', 'S3c7.csv']
    alcohol_concentrations = [0, 71, 141]
    analyze_single_elements(alcohol_files, alcohol_concentrations, 'Alcohol')

    # Combination analysis
    combination_files = ['S3c8.csv', 'S3c9.csv', 'S3c10.csv', 'S3c11.csv', 'S3c12.csv', 'S3c13.csv', 'S3c14.csv']
    combination_conditions = [
        {'S': 150, 'N': 0.25, 'A': 71},
        {'S': 300, 'N': 0.25, 'A': 71},
        {'S': 0, 'N': 0.25, 'A': 71},
        {'S': 150, 'N': 0.5, 'A': 71},
        {'S': 150, 'N': 0, 'A': 71},
        {'S': 150, 'N': 0.25, 'A': 141},
        {'S': 150, 'N': 0.25, 'A': 0}
    ]
    analyze_combinations(combination_files, combination_conditions)

    # Additional test points
    additional_files = ['S3c15.csv', 'S3c16.csv', 'S3c17.csv', 'S3c18.csv']
    additional_conditions = [
        {'S': 180, 'N': 0.4, 'A': 0},
        {'S': 260, 'N': 0.1, 'A': 0},
        {'S': 130, 'N': 0, 'A': 61},
        {'S': 0, 'N': 0, 'A': 122}
    ]
    analyze_combinations(additional_files, additional_conditions)
    
    """
    """
    print("A: Single Element Analysis")
    sugar_files = ['S3demine.csv', 'S3c2.csv', 'S3c3.csv']
    sugar_concentrations = [0, 150, 300]
    analyze_single_elements(sugar_files, sugar_concentrations, 'Sugar')

    nitrogen_files = ['S3demine.csv', 'S3c4.csv', 'S3c5.csv']
    nitrogen_concentrations = [0, 0.25, 0.5]
    analyze_single_elements(nitrogen_files, nitrogen_concentrations, 'Nitrogen')

    alcohol_files = ['S3demine.csv', 'S3c6.csv', 'S3c7.csv']
    alcohol_concentrations = [0, 71, 141]
    analyze_single_elements(alcohol_files, alcohol_concentrations, 'Alcohol')

    print("\nB: Combination Analysis")
    combination_files = ['S3c8.csv', 'S3c9.csv', 'S3c10.csv', 'S3c11.csv', 'S3c12.csv', 'S3c13.csv', 'S3c14.csv']
    combination_conditions = [
        {'S': 150, 'N': 0.25, 'A': 71},  # S3c8 (base)
        {'S': 300, 'N': 0.25, 'A': 71},  # S3c9
        {'S': 0, 'N': 0.25, 'A': 71},    # S3c10
        {'S': 150, 'N': 0.5, 'A': 71},   # S3c11
        {'S': 150, 'N': 0, 'A': 71},     # S3c12
        {'S': 150, 'N': 0.25, 'A': 141}, # S3c13
        {'S': 150, 'N': 0.25, 'A': 0}    # S3c14
    ]
    analyze_combinations(combination_files, combination_conditions)

    print("\nC: Sensitivity Analysis")
    sensitivity_analysis()

    print("\nD: Fermentation Steps")
    fermentation_steps()
    """
    #plot_multi_nyquist(Sugar_files,phase_shift=False)
    #calculate_and_plot_bode_multi(Sugar_files,phase_shift=False)
    #multidatapath = 'C:/Users/user/Downloads/Fermentation data and Bode analysis/IMPORTANT/MULTI_PARAMETERS_MEAN/'
    #impedance_vs_concentration_new(multidatapath)
    #plot_3d_impedance_spectra('FERMENTATION.csv')       
    #plot_impedance_analysis('RC.csv')
    #plot_multiple_impedance_analyses(S1OnlySugarfile)
    #plot_and_fit_impedance('RC.csv',order=1,time_point=0)
    #plot_and_fit_impedance('RC.csv',order=2,time_point=0)
    #plot_and_fit_impedance('RC.csv',order=3,time_point=0)
    #plot_and_fit_impedance("RC.csv",order=0,time_point=0,smooth=False)
    #Sugar_files = ['sucre_60_18.csv', 'sucre_120_18.csv', 'sucre_180_18.csv', 'sucre_240_18.csv', 'sucre_300_18.csv']
    #Alcool_files =['alcool30.csv','alcool60.csv','alcool90.csv','alcool120.csv','alcool140.csv']
    #concentrations, cutoff_freqs_sugar = analyze_cutoff_frequency_evolution(Sugar_files, 'Sugar')
    #concentrations, cutoff_freqs_Alcool = analyze_cutoff_frequency_evolution(Alcool_files, 'Alcool')
    """
    print("\nFitting Analysis for Different Combinations:")
    
    files = ['S3demine.csv'] + [f'S3c{i}.csv' for i in range(8, 15)]
    conditions = [
        {'S': 150, 'N': 0.25, 'A': 71},  # S3c8 (base)
        {'S': 300, 'N': 0.25, 'A': 71},  # S3c9
        {'S': 0, 'N': 0.25, 'A': 71},    # S3c10
        {'S': 150, 'N': 0.5, 'A': 71},   # S3c11
        {'S': 150, 'N': 0, 'A': 71},     # S3c12
        {'S': 150, 'N': 0.25, 'A': 141}, # S3c13
        {'S': 150, 'N': 0.25, 'A': 0},   # S3c14
    ]

    all_data = []

    for file, condition in zip(files, conditions):
        print(f"\nAnalyzing {file}: S:{condition['S']}, N:{condition['N']}, A:{condition['A']}")
        for order in [0,1,2,3]:  # You can add more orders if needed
            circuit, data, frequencies = plot_and_fit_impedance(file, order=order, time_point=0, phase_shift=True, smooth=False, optimize=True)
            
            # Add condition information
            data['Sugar'] = condition['S']
            data['Nitrogen'] = condition['N']
            data['Alcohol'] = condition['A']
            # Add initial guesses and fitted parameters
            param_names = circuit.get_param_names()
            initial_guesses = circuit.initial_guess
            fitted_params = circuit.parameters_
            
            for param, initial, fitted in zip(param_names, initial_guesses, fitted_params):
                data[f'{param}_initial'] = initial
                data[f'{param}_fitted'] = fitted
                
            all_data.append(data)

       # Write data to Excel
    write_to_excel(all_data, "impedance_analysis_results_S3demine_to_S3c14.xlsx")
    """


print("\nComparing Combined Solution with Individual Components:")

compare_impedances('S3c2.csv', 'S3c6.csv', 'S3c12.csv', 
                   {'measured': 'Measured (S3c12)', 
                    'calculated': 'Calculated (S3c2 + S3c6)'})

print("\nComparing Combined Solution with Individual Components (RC Parallel Model):")
fit_and_compare_rc_circuits('S3c4.csv', 'S3c6.csv', 'S3c10.csv', 
                            {'file1': 'S3c2', 'file2': 'S3c4', 'file_combined': 'S3c12'})

"""
    orders = [0, 1, 2, 3]
    smooth_options = [False, True]
    all_data = []

    for file in files:
        for order in orders:
            for smooth in smooth_options:
                circuit, data = plot_and_fit_impedance(file, order=order, time_point=0, phase_shift=False, smooth=smooth)
                all_data.append(data)

    # Write data to Excel
    write_to_excel(all_data)
    
"""

################################################################################################      
###---------------------------------Main Functions-----------------------------------####
################################################################################################   
 
def time_monitoring(file, CO2_on_off):  # main function to display impedance and CO2 release velocity as a function of time
    # file = 'FERMENTATION.csv'
    # file = 'gazeux2H.csv'
    freq_to_study = [1000]
    averaging_window = 1
    complex_study = 'yes'  # yes to activate complex conversion before filtering
    imp, phase, frequencies, studied_frequencies = duration_monitoring(file, freq_to_study)
    print("imp  :", imp)
    print("phase:", phase)
    impedance, time = impedance_vs_hours_by_complex(imp, phase, studied_frequencies, averaging_window, complex_study)
    # plot_CO2_velocity('R1.txt')
    
    if CO2_on_off == 1:
         impedance, time = plot_impedance_and_velocity('R1.csv', imp, phase, studied_frequencies, averaging_window, complex_study)
    else:
         impedance, time = impedance_vs_hours_by_complex(imp, phase, studied_frequencies, averaging_window, complex_study)
        
    # csv_for_scilab("monitoring_at_40kHz", impedance, time)
    # imp, phase, frequencies, studied_frequencies = duration_monitoring(file, freq_to_study)
    # write_to_csv(imp, 'data40kHz.csv')

# ----------------------------------------------------------------------------------------------
def averages(folder):  # plot impedance curves (magnitude and phase) vs frequency for all files in a folder
    # folder = input('Folder containing measurements: ')      
    # folder = 'test_agitation_carte4_averages'
    # folder = 'test_gaz'
    # folder = 'sucre_18janvier'
    means_by_complex_folder(folder)  # average by converting to complex
    # print_folder_averages(folder)

# ----------------------------------------------------------------------------------------------
def csv_averages(folder):
    print("--- Enter averages of measurements in a folder into a new csv file ---")
    # folder = input('Folder containing measurements: ')
    # folder = 'sucre_18janvier'
    # folder = '8910'
    avg_magnitude, avg_phase, frequencies = means_by_complex_folder(folder)
    print(avg_magnitude)
    print("Number of files:", len(avg_magnitude) - 1)
    filename = input('csv name for scilab: ')          
    csv_for_scilab(filename, avg_magnitude, frequencies)

if __name__ == "__main__":  
    main()    