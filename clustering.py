# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
from database import get_cluster_center_from_selectedrows
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from filter_table import filter_rows_by_conf_instr
from database import get_cluster_center
from sklearn.metrics import r2_score
import numpy as np
import hdbscan
import seaborn as sns
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit



def parabel(x, a, b, c):
    return a * x**2 + b * x + c

def kubisch(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def sinus(x, a, b, c):
    return a * np.sin(b * x + c)

def linear(x, a, b):
    return a * x + b

def cosinus(x, a, b, c):
    return a * np.cos(b * x + c)

def exponential(x, a, b):
    return a * np.exp(b * x)

def ransac_poly_fit(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), RANSACRegressor())
    model.fit(x[:, np.newaxis], y)
    return model

def calculate_r2(model, x_data, y_data):
    predictions = model.predict(x_data)
    return r2_score(y_data, predictions)

def calculate_adjusted_r_squared(model, x_data, y_data, num_predictors):
    r2 = calculate_r2(model, x_data, y_data)
    n = len(y_data)  # Anzahl der Beobachtungen
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)
    return adj_r2

# Sinus- und Cosinus-Transformationen
def sinus_cosinus_transform(x):
    return np.column_stack((np.sin(x), np.cos(x)))

def exponential_transform(x):
    x = np.clip(x, a_min=0, a_max=None)  # Verhindert negative Eingaben für den Logarithmus
    return np.log1p(x)  # Logarithmus-Transformation

def ransac_nonlinear_fit(x, y, transform_func):
    transformer = FunctionTransformer(transform_func, validate=True)
    model = make_pipeline(transformer, RANSACRegressor())
    model.fit(x[:, np.newaxis], y)
    return model

def sinus_transform(x):
    return np.sin(x)

def cosinus_transform(x):
    return np.cos(x)

def calculate_adjusted_r_squared(model, x_data, y_data, num_predictors):
    predictions = model.predict(x_data)
    r2 = r2_score(y_data, predictions)
    n = len(y_data)  # Anzahl der Beobachtungen
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_predictors - 1)
    return adj_r2


def create_ransac_pipeline(func, degree=1, transformer=None):
    if transformer:
        return make_pipeline(FunctionTransformer(transformer, validate=False), 
                             PolynomialFeatures(degree=degree), RANSACRegressor())
    else:
        return make_pipeline(PolynomialFeatures(degree=degree), RANSACRegressor())

def get_poly_model_parameters(model, degree):
    estimator = model.named_steps['ransacregressor'].estimator_
    coefs = estimator.coef_
    intercept = estimator.intercept_

    # Für Polyfunktionen ist der Koeffizient für den Term x^0 (also das Absolutglied) im Intercept
    return np.concatenate(([intercept], coefs[1:degree+1])) 

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
                    popt_sinus, _ = curve_fit(sinus, x_data, y_data)
                    popt_linear, _ = curve_fit(linear, x_data, y_data)

                    # Angepasste Funktionen berechnen
                    y_parabel_fit = parabel(x_data, *popt_parabel)
                    y_kubisch_fit = kubisch(x_data, *popt_kubisch)
                    y_sinus_fit = sinus(x_data, *popt_sinus)
                    y_linear_fit = linear(x_data, *popt_linear)
                    
                    poly_eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
                    sinus_eq = f'{popt_sinus[0]:.2f} * sin({popt_sinus[1]:.2f} * x + {popt_sinus[2]:.2f})'
                    kubisch_eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
                    linear_eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'

                    xy = np.vstack([df['wall_thickness'], df['magnetization']])
                    kde = gaussian_kde(xy)(xy)
                    #norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                    #sizes = (1 - norm_velocity) * 50

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, edgecolor='none', cmap='coolwarm')
                    plt.plot(x_data, y_parabel_fit, label='Parabel-Fit', color='red')
                    plt.plot(x_data, y_kubisch_fit, label='Kubisch-Fit', color='orange')
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
    x_data = np.array(df['x_coord'].tolist()).reshape(-1, 1)
    y_data = np.array(df['y_coord'].tolist())
    model_parabel = create_ransac_pipeline(None, degree=2)
    model_kubisch = create_ransac_pipeline(None, degree=3)
    model_sinus = create_ransac_pipeline(sinus_transform, degree=1)  # Definiere sinus_transform entsprechend
    model_cosinus = create_ransac_pipeline(cosinus_transform, degree=1)  # Definiere cosinus_transform entsprechend
    model_linear = create_ransac_pipeline(None, degree=1)
    model_exponential = create_ransac_pipeline(exponential_transform, degree=1)  # Definiere exponential_transform entsprechend


    # Anpassen der Modelle
    model_parabel.fit(x_data, y_data)
    model_kubisch.fit(x_data, y_data)
    model_sinus.fit(x_data, y_data)
    model_cosinus.fit(x_data, y_data)
    model_linear.fit(x_data, y_data)
    model_exponential.fit(x_data, y_data)
    
    num_predictors_parabel = 3  # a, b, c
    num_predictors_kubisch = 4  # a, b, c, d
    num_predictors_sinus = 3  # a, b, c
    num_predictors_linear = 2  # a, b
    num_predictors_cosinus = 3  # a, b, c
    num_predictors_exponential = 2  # a, b

    adj_r2_parabel = calculate_adjusted_r_squared(model_parabel, x_data, y_data, num_predictors_parabel)
    adj_r2_kubisch = calculate_adjusted_r_squared(model_kubisch, x_data, y_data, num_predictors_kubisch)
    adj_r2_sinus = calculate_adjusted_r_squared(model_sinus, x_data, y_data, num_predictors_sinus)
    adj_r2_linear = calculate_adjusted_r_squared(model_linear, x_data, y_data, num_predictors_linear)
    adj_r2_cosinus = calculate_adjusted_r_squared(model_cosinus, x_data, y_data, num_predictors_cosinus)
    adj_r2_exponential = calculate_adjusted_r_squared(model_exponential, x_data, y_data, num_predictors_exponential)

    r2_values = {
    'Parabel': calculate_adjusted_r_squared(model_parabel, x_data, y_data, num_predictors_parabel),
    'Kubisch': calculate_adjusted_r_squared(model_kubisch, x_data, y_data, num_predictors_kubisch),
    'Sinus': calculate_adjusted_r_squared(model_sinus, x_data, y_data, num_predictors_sinus),
    'Linear': calculate_adjusted_r_squared(model_linear, x_data, y_data, num_predictors_linear),
    'Cosinus': calculate_adjusted_r_squared(model_cosinus, x_data, y_data, num_predictors_cosinus),
    'Exponential': calculate_adjusted_r_squared(model_exponential, x_data, y_data, num_predictors_exponential)
    }
    print(r2_values)
    best_two = sorted(r2_values, key=r2_values.get, reverse=True)[:2]


    print("Adjusted R-squared for Parabel: ", adj_r2_parabel)
    print("Adjusted R-squared for Kubisch: ", adj_r2_kubisch)
    print("Adjusted R-squared for Sinus: ", adj_r2_sinus)
    print("Adjusted R-squared for Linear: ", adj_r2_linear)
    print("Adjusted R-squared for Cosinus: ", adj_r2_cosinus)
    print("Adjusted R-squared for Exponential: ", adj_r2_exponential)

    parabel_params = get_poly_model_parameters(model_parabel, 2) # Grad 2 für Parabel
    kubisch_params = get_poly_model_parameters(model_kubisch, 3) # Grad 3 für Kubisch
    linear_params = get_poly_model_parameters(model_linear, 1)   # Grad 1 für Linear

    parabel_eq = f"{parabel_params[2]:.2f}x² + {parabel_params[1]:.2f}x + {parabel_params[0]:.2f}"

    # Kubisch
    kubisch_eq = f"{kubisch_params[3]:.2f}x³ + {kubisch_params[2]:.2f}x² + {kubisch_params[1]:.2f}x + {kubisch_params[0]:.2f}"

    # Linear
    linear_eq = f"{linear_params[1]:.2f}x + {linear_params[0]:.2f}"



    #if len(x_data) > 0 and len(y_data) > 0:
     #   fig, ax = plt.subplots()
    """
    popt_parabel, _ = curve_fit(parabel, x_data, y_data)
    popt_kubisch, _ = curve_fit(kubisch, x_data, y_data, maxfev=10000)
    popt_sinus, _ = curve_fit(sinus, x_data, y_data)
    popt_linear, _ = curve_fit(linear, x_data, y_data)
    popt_cosinus, _ = curve_fit(cosinus, x_data, y_data)
    popt_exponential, _ = curve_fit(exponential, x_data, y_data)

    # Angepasste Funktionen berechnen
    y_parabel_fit = parabel(x_data, *popt_parabel)
    y_kubisch_fit = kubisch(x_data, *popt_kubisch)
    y_sinus_fit = sinus(x_data, *popt_sinus)
    y_linear_fit = linear(x_data, *popt_linear)
    y_cosinus_fit = cosinus(x_data, *popt_cosinus)
    y_exponential_fit = exponential(x_data, *popt_exponential)
    num_predictors_parabel = 3  # a, b, c
    num_predictors_kubisch = 4  # a, b, c, d
    num_predictors_sinus = 3  # a, b, c
    num_predictors_linear = 2  # a, b
    num_predictors_cosinus = 3  # a, b, c
    num_predictors_exponential = 2  # a, b

    # Berechnung des angepassten R-Quadrat für jede Funktion
    adj_r2_parabel = calculate_adjusted_r_squared(y_parabel_fit, x_data, y_data, num_predictors_parabel)
    adj_r2_kubisch = calculate_adjusted_r_squared(y_kubisch_fit, x_data, y_data, num_predictors_kubisch)
    adj_r2_sinus = calculate_adjusted_r_squared(y_sinus_fit, x_data, y_data, num_predictors_sinus)
    adj_r2_linear = calculate_adjusted_r_squared(y_linear_fit, x_data, y_data, num_predictors_linear)
    adj_r2_cosinus = calculate_adjusted_r_squared(y_cosinus_fit, x_data, y_data, num_predictors_cosinus)
    adj_r2_exponential = calculate_adjusted_r_squared(y_exponential_fit, x_data, y_data, num_predictors_exponential)

    #Print the adjusted R-squared values
    print("Adjusted R-squared for Parabel: ", adj_r2_parabel)
    print("Adjusted R-squared for Kubisch: ", adj_r2_kubisch)
    print("Adjusted R-squared for Linear: ", adj_r2_linear)
    print("Adjusted R-squared for Sinus: ", adj_r2_sinus)
    print("Adjusted R-squared for Cosinus: ", adj_r2_cosinus)
    print("Adjusted R-squared for Exponential: ", adj_r2_exponential)

    
    # Formatierung der Gleichungen für alle Fits
    poly_eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
    sinus_eq = f'{popt_sinus[0]:.2f} * sin({popt_sinus[1]:.2f} * x + {popt_sinus[2]:.2f})'
    kubisch_eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
    linear_eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'
    cosinus_eq = f'{popt_cosinus[0]:.2f} * cos({popt_cosinus[1]:.2f} * x + {popt_cosinus[2]:.2f})'
    exp_eq = f'{popt_exponential[0]:.2f} * exp({popt_exponential[1]:.2f} * x)'  
    # Erstellen deines Plots
    """
    fig, ax = plt.subplots()
    #plt.scatter(x_data, y_data, label='Originaldaten')
    plt.scatter(df['x_coord'], df['y_coord'], label='Originaldaten')
    """
    plt.plot(x_data, y_parabel_fit, label='Parabel-Fit', color='red')
    plt.plot(x_data, y_sinus_fit, label='Sinus-Fit', color='green')
    plt.plot(x_data, y_kubisch_fit, label='Kubisch-Fit', color='orange')
    plt.plot(x_data, y_linear_fit, label='Linear-Fit', color='purple')
    plt.plot(x_data, y_cosinus_fit, label='Cosinus-Fit', color='blue')
    plt.plot(x_data, y_exponential_fit, label='Exponential-Fit', color='magenta')
    """

    x_range = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
    if 'Parabel' in best_two:
        plt.plot(x_range, model_parabel.predict(x_range), label='Beste Parabel-Fit', color='red')
    if 'Kubisch' in best_two:
        plt.plot(x_range, model_kubisch.predict(x_range), label='Beste Kubisch-Fit', color='orange')
    if 'Sinus' in best_two:
        plt.plot(x_range, model_sinus.predict(x_range), label='Beste Sinus-Fit', color='green')
    if 'Linear' in best_two:
        plt.plot(x_range, model_linear.predict(x_range), label='Beste Linear-Fit', color='purple')
    if 'Cosinus' in best_two:
        plt.plot(x_range, model_cosinus.predict(x_range), label='Beste Cosinus-Fit', color='blue')
    if 'Exponential' in best_two:
        plt.plot(x_range, model_exponential.predict(x_range), label='Beste Exponential-Fit', color='magenta')


    """
        # Füge die Gleichungen zum Plot hinzu
    plt.legend()
    fig.text(0.05, 0.02, f'Parabel: {poly_eq}', fontsize=10)
    fig.text(0.05, 0.01, f'Sinus: {sinus_eq}', fontsize=10)
    fig.text(0.95, 0.02, f'Kubisch: {kubisch_eq}', fontsize=10, ha='right')
    fig.text(0.95, 0.00, f'Linear: {linear_eq}', fontsize=10, ha='right')
    fig.text(0.95, -0.05, f'Cosinus: {cosinus_eq}', fontsize=10, ha='right')
    fig.text(0.95, -0.10, f'Exponential: {exp_eq}', fontsize=10, ha='right')

    """
    plt.xlabel('Magnetization')
    plt.ylabel('Wall Thickness')
    plt.legend()
    plt.show()

