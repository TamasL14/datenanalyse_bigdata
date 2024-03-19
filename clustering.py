# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
from database import get_cluster_center_from_selectedrows
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from filter_table import filter_rows_by_conf_instr
from database import get_cluster_center
import numpy as np
import hdbscan
import seaborn as sns
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit


# Funktionen für die polynomiale Transformation
def poly_features(degree):
    return PolynomialFeatures(degree=degree, include_bias=False)

# Robustes Fitting-Modell erstellen
def ransac_fit(x, y, degree):
    model = make_pipeline(poly_features(degree), RANSACRegressor())
    model.fit(x[:, np.newaxis], y)
    return model

def parabel(x, a, b, c):
    return a * x**2 + b * x + c

def kubisch(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def abs_value(x, a, b, c):
    return a * np.abs(x) + b * x + c

def sinus(x, a, b, c):
    return a * np.sin(b * x + c)

def linear(x, a, b):
    return a * x + b

# Definition des Clustering Fensters
def clustering_window():
    layout = [
        [sg.Text("Wähle bitte zwei Eigenschaften für die DBScan Clustering Analyse", justification="center")],
        [sg.Column([[sg.Check('Defect Channel', key='defect_channel'), sg.Check('Distance', key='distance'), sg.Check('Magnetization', key='magnetization'), sg.Check('Timestamp', key='timestamp'), sg.Check('Velocity', key='velocity'), sg.Check('Wall Thickness', key='wall_thickness')]], justification="center")],
        [sg.Text("Wähle bitte die Anzahl der Cluster",visible=False), sg.InputText(key='k', visible=False)],
        [sg.Text("Wähle bitte min_samples",visible=False), sg.InputText(key='min_samples',visible=False)],
        [sg.Column([[sg.Button("Submit", key='-SUBMIT-'), sg.Button("Cancel", key='-CANCEL-')]], justification="center")],
    ]
    # Fenster erstellen
    return sg.Window("Clustering", layout, size=(600, 400))



# Funktion zum Öffnen des Clustering Fensters
def clustering(selected_rows):
    while True:
        window = clustering_window() # Öffnen des Clustering Fensters
        event, values = window.read() # Benutzerinteraktion mit dem Fenster lesen
        # Wenn das Fenster geschlossen wird oder der "Cancel" Button gedrückt wird
        if event == '-CANCEL-' or event == sg.WIN_CLOSED: 
            window.close() # Fenster schließen
            break

        # Wenn der "Submit" Button gedrückt wird
        if event == '-SUBMIT-':

            # Variablen initialisieren
            punkte=[]
            plotting={}
            property_list=[element for element in values if values[element]==True] # Ausgewählte Eigenschaften in einer Liste speichern
        
            if len(property_list)>0: # Überprüfen, ob mindestens eine Eigenschaft ausgewählt wurde
                df = pd.DataFrame(columns=['data_id', 'magnetization', 'velocity', 'wall_thickness'])
                for data_id in selected_rows: # Für jede ausgewählte Zeile
                    punkte=get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                    plotting[data_id]=punkte # Datenpunkte in ein Dictionary speichern

                if set(property_list) == {'magnetization', 'velocity', 'wall_thickness'}:
                    for data_id in plotting: 
                        # Assuming data structure: [[magnetization], [velocity], [wall_thickness]]
                        daten = plotting[data_id]
                        temp_df = pd.DataFrame(daten).transpose()
                        temp_df.columns = property_list
                        temp_df['data_id'] = data_id
                        df = pd.concat([df, temp_df], ignore_index=True)
                    
                    x_data = df['wall_thickness']
                    y_data = df['magnetization']
                    
                    popt_parabel, _ = curve_fit(parabel, x_data, y_data)
                    popt_kubisch, _ = curve_fit(kubisch, x_data, y_data, maxfev=10000)
                    popt_abs, _ = curve_fit(abs_value, x_data, y_data, maxfev=10000)
                    popt_sinus, _ = curve_fit(sinus, x_data, y_data)
                    popt_linear, _ = curve_fit(linear, x_data, y_data)

                    # Angepasste Funktionen berechnen
                    y_parabel_fit = parabel(x_data, *popt_parabel)
                    y_kubisch_fit = kubisch(x_data, *popt_kubisch)
                    y_abs_fit = abs_value(x_data, *popt_abs)
                    y_sinus_fit = sinus(x_data, *popt_sinus)
                    y_linear_fit = linear(x_data, *popt_linear)
                    
                    poly_eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
                    sinus_eq = f'{popt_sinus[0]:.2f} * sin({popt_sinus[1]:.2f} * x + {popt_sinus[2]:.2f})'
                    kubisch_eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
                    abs_eq = f'{popt_abs[0]:.2f} * abs(x) + {popt_abs[1]:.2f} * x + {popt_abs[2]:.2f}'
                    linear_eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'

                    xy = np.vstack([df['wall_thickness'], df['magnetization']])
                    kde = gaussian_kde(xy)(xy)
                    #norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                    #sizes = (1 - norm_velocity) * 50

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, edgecolor='none', cmap='coolwarm')
                    plt.plot(x_data, y_parabel_fit, label='Parabel-Fit', color='red')
                    plt.plot(x_data, y_kubisch_fit, label='Kubisch-Fit', color='orange')
                    plt.plot(x_data, y_abs_fit, label='Absolutwert-Fit', color='yellow')
                    plt.plot(x_data, y_sinus_fit, label='Sinus-Fit', color='green')
                    plt.plot(x_data, y_linear_fit, label='Linear-Fit', color='blue')
                    
                    
                    
                    fig.colorbar(scatter, ax=ax, label='Density')
                    plt.ylabel('Magnetization')
                    plt.xlabel('Wall Thickness')    
                    try:
                        plt.show(block=True) 
                        plt.close()
                    except:
                        pass  
                else:
                    for data_id in plotting: # Für jede ausgewählte Zeile
                        for punkte  in plotting[data_id]: # Für jeden Datenpunkt
                            plt.plot(punkte,'o') # Datenpunkte plotten
                        plt.legend(property_list,loc='upper center') # Legende hinzufügen
                        try:
                            plt.show(block=True) # Plot anzeigen
                            plt.close() # Plot schließen
                        except:
                            pass
            else:
                # Fehlermeldung, wenn keine Eigenschaft ausgewählt wurde
                sg.popup_quick_message("Please select at least one property", auto_close=True, auto_close_duration=2)
        
        continue # Nächste Benutzerinteraktion lesen


def cluster_hdbscan(selected_rows):
    while True:
        window = clustering_window() # Öffnen des Clustering Fensters
        event, values = window.read() # Benutzerinteraktion mit dem Fenster lesen
        # Wenn das Fenster geschlossen wird oder der "Cancel" Button gedrückt wird
        if event == '-CANCEL-' or event == sg.WIN_CLOSED: 
            window.close() # Fenster schließen
            break

        # Wenn der "Submit" Button gedrückt wird
        if event == '-SUBMIT-':

            # Variablen initialisieren
            punkte=[]
            plotting={}
            property_list=[element for element in values if values[element]==True] # Ausgewählte Eigenschaften in einer Liste speichern
            magnetization = []
            wall_thickness = []
            
            if len(property_list)>0: # Überprüfen, ob mindestens eine Eigenschaft ausgewählt wurde
                for data_id in selected_rows: # Für jede ausgewählte Zeile
                    punkte=get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                    plotting[data_id]=punkte # Datenpunkte in ein Dictionary speichern

                if set(property_list) == {'magnetization', 'wall_thickness'}:
                    for data_id in plotting: 
                        # Assuming data structure: [[magnetization], [velocity], [wall_thickness]]
                        magnetization.extend(plotting[data_id][property_list.index('magnetization')])
                        wall_thickness.extend(plotting[data_id][property_list.index('wall_thickness')])
                        
                        combined_array = np.column_stack((magnetization, wall_thickness))

                        clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(combined_array)
                        cluster_labels = clusterer.labels_
                        # Berechne die Mittelpunkte für jeden Cluster
                        unique_clusters = set(cluster_labels)
                        cluster_centers = []

                        for cluster in unique_clusters:
                            if cluster != -1:  # Ignoriere Rauschen, das als -1 gekennzeichnet ist
                                points_in_cluster = combined_array[cluster_labels == cluster]
                                cluster_center = points_in_cluster.mean(axis=0)
                                cluster_centers.append(cluster_center)
                        print(cluster_centers)
                        cluster_colors = [plt.cm.Spectral(each)
                        for each in np.linspace(0, 1, len(set(cluster_labels)))]

                        for i, color in enumerate(cluster_colors):
                            mask = (cluster_labels == i)
                            if np.any(mask):
                                plt.scatter(np.array(magnetization)[mask], np.array(wall_thickness)[mask], s=50, linewidth=0, c=[color], alpha=0.5, label=f'Cluster {i}')

                    plt.xlabel('Magnetization')
                    plt.ylabel('Wall Thickness')
                    plt.show(block=True) 
                    plt.close()
                else:
                    for data_id in plotting: # Für jede ausgewählte Zeile
                        for punkte  in plotting[data_id]: # Für jeden Datenpunkt
                            plt.plot(punkte,'o') # Datenpunkte plotten
                        plt.legend(property_list,loc='upper center') # Legende hinzufügen
                        try:
                            plt.show(block=True) # Plot anzeigen
                            plt.close() # Plot schließen
                        except:
                            pass
            else:
                # Fehlermeldung, wenn keine Eigenschaft ausgewählt wurde
                sg.popup_quick_message("Please select at least one property", auto_close=True, auto_close_duration=2)
        
        continue # Nächste Benutzerinteraktion lesen

"""
# Standard plotall
def plotall(selected_rows):
    property_list = ['magnetization', 'wall_thickness'] 
    configuration = ['Europe', 'Australia', 'Africa', 'America', 'Africa']
    instruments = ['Dolphin', 'Pufferfish', 'Unicorn', 'Dog', 'Elephant']
    instruments = ['Dolphin']
    configuration = ['Africa']
    # Anzahl der Subplots bestimmen
    num_configs = len(configuration)
    num_instrs = len(instruments)
    fig, axs = plt.subplots(num_configs, num_instrs, figsize=(5*num_instrs, 5*num_configs))
    for i, conf in enumerate(configuration):
        for j, instr in enumerate(instruments):
            df = pd.DataFrame(columns=['data_id', 'center'])
            df = get_cluster_center(instr, conf)
            print(df)
            x_data = []
            y_data = []

            for coord in df:
                if len(coord) == 2:  # Stelle sicher, dass es sich um ein Koordinatenpaar handelt
                    x_data.append(coord[0])
                    y_data.append(coord[1])

            # Überprüfe, ob Daten vorhanden sind
            if x_data and y_data:
                x_data = np.array(x_data).ravel().astype(float)
                y_data = np.array(y_data).ravel().astype(float)        

            popt_parabel, _ = curve_fit(parabel, x_data, y_data)
            popt_kubisch, _ = curve_fit(kubisch, x_data, y_data, maxfev=10000)
            popt_abs, _ = curve_fit(abs_value, x_data, y_data, maxfev=10000)
            popt_sinus, _ = curve_fit(sinus, x_data, y_data)
            popt_linear, _ = curve_fit(linear, x_data, y_data)

            # Angepasste Funktionen berechnen
            y_parabel_fit = parabel(x_data, *popt_parabel)
            y_kubisch_fit = kubisch(x_data, *popt_kubisch)
            y_abs_fit = abs_value(x_data, *popt_abs)
            y_sinus_fit = sinus(x_data, *popt_sinus)
            y_linear_fit = linear(x_data, *popt_linear)
            
            # Formatierung der Gleichungen für alle Fits
            poly_eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
            sinus_eq = f'{popt_sinus[0]:.2f} * sin({popt_sinus[1]:.2f} * x + {popt_sinus[2]:.2f})'
            kubisch_eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
            abs_eq = f'{popt_abs[0]:.2f} * abs(x) + {popt_abs[1]:.2f} * x + {popt_abs[2]:.2f}'
            linear_eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'

            # Erstellen deines Plots
            fig, ax = plt.subplots()
            plt.scatter(x_data, y_data, label='Originaldaten')
            plt.plot(x_data, y_parabel_fit, label='Parabel-Fit', color='red')
            plt.plot(x_data, y_sinus_fit, label='Sinus-Fit', color='green')
            plt.plot(x_data, y_kubisch_fit, label='Kubisch-Fit', color='orange')
            plt.plot(x_data, y_abs_fit, label='Absolutwert-Fit', color='yellow')
            plt.plot(x_data, y_linear_fit, label='Linear-Fit', color='purple')

            # Füge die Gleichungen zum Plot hinzu
            plt.legend()
            fig.text(0.05, 0.02, f'Parabel: {poly_eq}', fontsize=10)
            fig.text(0.05, 0.01, f'Sinus: {sinus_eq}', fontsize=10)
            fig.text(0.95, 0.02, f'Kubisch: {kubisch_eq}', fontsize=10, ha='right')
            fig.text(0.95, 0.01, f'Absolutwert: {abs_eq}', fontsize=10, ha='right')
            fig.text(0.95, 0.00, f'Linear: {linear_eq}', fontsize=10, ha='right')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Vergleich verschiedener Fits')          
    plt.show()
    """
"""
def plotall(selected_rows):


    df = pd.DataFrame(columns=['data_id', 'center'])
    for data_id in selected_rows:  # Für jede ausgewählte Zeile
        temp_data = get_cluster_center_from_selectedrows(data_id)
        # Überprüfe, ob temp_data ein DataFrame ist, und konvertiere, falls nötig
        if isinstance(temp_data, list):
            temp_df = pd.DataFrame(temp_data, columns=['x_coord', 'y_coord'])
        else:
            temp_df = temp_data
        df = pd.concat([df, temp_df], ignore_index=True)

    x_data = df['x_coord'].tolist()
    y_data = df['y_coord'].tolist()

    for coord in df:
        if len(coord) == 2:  # Stelle sicher, dass es sich um ein Koordinatenpaar handelt
            x_data.append(coord[0])
            y_data.append(coord[1])

    # Überprüfe, ob Daten vorhanden sind
    if x_data and y_data:
        x_data = np.array(x_data).ravel().astype(float)
        y_data = np.array(y_data).ravel().astype(float)        

    popt_parabel, _ = curve_fit(parabel, x_data, y_data)
    popt_kubisch, _ = curve_fit(kubisch, x_data, y_data, maxfev=10000)
    popt_abs, _ = curve_fit(abs_value, x_data, y_data, maxfev=10000)
    popt_sinus, _ = curve_fit(sinus, x_data, y_data)
    popt_linear, _ = curve_fit(linear, x_data, y_data)

    # Angepasste Funktionen berechnen
    y_parabel_fit = parabel(x_data, *popt_parabel)
    y_kubisch_fit = kubisch(x_data, *popt_kubisch)
    y_abs_fit = abs_value(x_data, *popt_abs)
    y_sinus_fit = sinus(x_data, *popt_sinus)
    y_linear_fit = linear(x_data, *popt_linear)
    
    # Formatierung der Gleichungen für alle Fits
    poly_eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
    sinus_eq = f'{popt_sinus[0]:.2f} * sin({popt_sinus[1]:.2f} * x + {popt_sinus[2]:.2f})'
    kubisch_eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
    abs_eq = f'{popt_abs[0]:.2f} * abs(x) + {popt_abs[1]:.2f} * x + {popt_abs[2]:.2f}'
    linear_eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'

    # Erstellen deines Plots
    fig, ax = plt.subplots()
    plt.scatter(x_data, y_data, label='Originaldaten')
    plt.plot(x_data, y_parabel_fit, label='Parabel-Fit', color='red')
    plt.plot(x_data, y_sinus_fit, label='Sinus-Fit', color='green')
    plt.plot(x_data, y_kubisch_fit, label='Kubisch-Fit', color='orange')
    plt.plot(x_data, y_abs_fit, label='Absolutwert-Fit', color='yellow')
    plt.plot(x_data, y_linear_fit, label='Linear-Fit', color='purple')

    # Füge die Gleichungen zum Plot hinzu
    plt.legend()
    fig.text(0.05, 0.02, f'Parabel: {poly_eq}', fontsize=10)
    fig.text(0.05, 0.01, f'Sinus: {sinus_eq}', fontsize=10)
    fig.text(0.95, 0.02, f'Kubisch: {kubisch_eq}', fontsize=10, ha='right')
    fig.text(0.95, 0.01, f'Absolutwert: {abs_eq}', fontsize=10, ha='right')
    fig.text(0.95, 0.00, f'Linear: {linear_eq}', fontsize=10, ha='right')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vergleich verschiedener Fits')
    plt.show()
"""

def plotall(selected_rows):
    df = pd.DataFrame(columns=['data_id', 'center'])
    for data_id in selected_rows:
        temp_data = get_cluster_center_from_selectedrows(data_id)
        if isinstance(temp_data, list):
            temp_df = pd.DataFrame(temp_data, columns=['y_coord', 'x_coord'])
        else:
            temp_df = temp_data
        df = pd.concat([df, temp_df], ignore_index=True)
        
    df = df.sort_values(by='x_coord', ascending=True)   
    x_data = np.array(df['x_coord'].tolist())
    y_data = np.array(df['y_coord'].tolist())

    if len(x_data) > 0 and len(y_data) > 0:
        fig, ax = plt.subplots()

        # Parabel (2. Grad)
        model_parabel = ransac_fit(x_data, y_data, 2)
        y_parabel_fit = model_parabel.predict(x_data[:, np.newaxis])
        plt.plot(x_data, y_parabel_fit, label='Robuster Parabel-Fit', color='red')

        # Kubisch (3. Grad)
        model_kubisch = ransac_fit(x_data, y_data, 3)
        y_kubisch_fit = model_kubisch.predict(x_data[:, np.newaxis])
        #plt.plot(x_data, y_kubisch_fit, label='Robuster Kubisch-Fit', color='orange')

        # Weitere Modelle (Absolutwert, Sinus, Linear) können hier hinzugefügt werden...

        plt.scatter(x_data, y_data, label='Originaldaten')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Vergleich verschiedener robuster Fits')
        plt.show()


