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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.fft import rfftfreq, rfft
from scipy import stats
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


def linear(x, a, b):
    return a * x + b

def sinus(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

def cosinus(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(frequency * x + phase) + offset

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_exponential(clustered_data_x, clustered_data_y):

    # Anfangsschätzungen für die Parameter a, b und c
    a_initial = np.max(clustered_data_y)
    b_initial = -1.0 / (np.max(clustered_data_x) - np.min(clustered_data_x))
    c_initial = np.min(clustered_data_y)

    # Anpassen der exponentiellen Funktion an die Daten
    popt_exponential, _ = curve_fit(exponential, clustered_data_x, clustered_data_y, 
                                    p0=[a_initial, b_initial, c_initial], maxfev=10000)
    return popt_exponential

def ransac_poly_fit(x, y, degree):
    model = make_pipeline(PolynomialFeatures(degree=degree), RANSACRegressor())
    model.fit(x[:, np.newaxis], y)
    return model

def calculate_r2(model, x_data, y_data):
    predictions = model.predict(x_data)
    return r2_score(y_data, predictions)

def calculate_adjusted_r2(r_squared, num_obs, num_predictors):
    adjusted_r2 = 1 - (1 - r_squared) * (num_obs - 1) / (num_obs - num_predictors - 1)
    return adjusted_r2

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

def calculate_adjusted_r_squared(y_true, y_pred, num_predictors):
    n = len(y_true)
    r_squared = r2_score(y_true, y_pred)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - num_predictors - 1)
    return adjusted_r_squared



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

#robustes fitting
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

    combined_array = np.column_stack((x_data, y_data))
    n_clusters = 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_array)
    cluster_labels = kmeans.labels_

    # Berechne die Distanzen zu den Clusterzentren
    centroids = kmeans.cluster_centers_
    distances = pairwise_distances(combined_array, centroids[cluster_labels], metric='euclidean')
    distances = np.min(distances, axis=1)  # Wähle die minimale Distanz für jeden Punkt

    # Bestimme den Schwellenwert anhand eines Prozentsatzes
    percentile = 95  # Zum Beispiel die 95. Perzentile
    threshold = np.percentile(distances, percentile)

    # Ausgabe des Schwellenwerts
    print(f"Schwellenwert für die {percentile}. Perzentile: {threshold}")

    # Optional: Identifiziere und entferne Ausreißer
    inliers = distances < threshold
    clustered_data_x = np.array(x_data[inliers]).reshape(-1, 1)

    clustered_data_y = np.array(y_data[inliers])

    # Teile Datenpunkte in Cluster auf
    clustered_data_x = [x_data_filtered[cluster_labels == i] for i in range(n_clusters)]
    clustered_data_y = [y_data_filtered[cluster_labels == i] for i in range(n_clusters)]



    clustered_data_x = [[] for _ in range(n_clusters)]
    clustered_data_y = [[] for _ in range(n_clusters)]
    
    # Teile Datenpunkte in die Listen basierend auf ihren Clustern auf
    for label, x, y in zip(cluster_labels, x_data, y_data):
        clustered_data_x[label].append(x)
        clustered_data_y[label].append(y)
    clustered_data_x = np.array(clustered_data_x[0]).reshape(-1, 1)
    clustered_data_y = np.array(clustered_data_y[0])


    model_parabel = create_ransac_pipeline(None, degree=2)
    model_kubisch = create_ransac_pipeline(None, degree=3)
    model_sinus = create_ransac_pipeline(sinus_transform, degree=1)  # Definiere sinus_transform entsprechend
    model_cosinus = create_ransac_pipeline(cosinus_transform, degree=1)  # Definiere cosinus_transform entsprechend
    model_linear = create_ransac_pipeline(None, degree=1)
    model_exponential = create_ransac_pipeline(exponential_transform, degree=1)  # Definiere exponential_transform entsprechend



    # Anpassen der Modelle
    model_parabel.fit(clustered_data_x, clustered_data_y)
    model_kubisch.fit(clustered_data_x, clustered_data_y)
    model_sinus.fit(clustered_data_x, clustered_data_y)
    model_cosinus.fit(clustered_data_x, clustered_data_y)
    model_linear.fit(clustered_data_x, clustered_data_y)
    model_exponential.fit(clustered_data_x, clustered_data_y)

    
    num_predictors_parabel = 3  # a, b, c
    num_predictors_kubisch = 4  # a, b, c, d
    num_predictors_sinus = 3  # a, b, c
    num_predictors_linear = 2  # a, b
    num_predictors_cosinus = 3  # a, b, c
    num_predictors_exponential = 2  # a, b

    adj_r2_parabel = calculate_adjusted_r_squared(model_parabel, clustered_data_x, clustered_data_y, num_predictors_parabel)
    adj_r2_kubisch = calculate_adjusted_r_squared(model_kubisch, clustered_data_x, clustered_data_y, num_predictors_kubisch)
    adj_r2_sinus = calculate_adjusted_r_squared(model_sinus, clustered_data_x, clustered_data_y, num_predictors_sinus)
    adj_r2_linear = calculate_adjusted_r_squared(model_linear, clustered_data_x, clustered_data_y, num_predictors_linear)
    adj_r2_cosinus = calculate_adjusted_r_squared(model_cosinus, clustered_data_x, clustered_data_y, num_predictors_cosinus)
    adj_r2_exponential = calculate_adjusted_r_squared(model_exponential, clustered_data_x, clustered_data_y, num_predictors_exponential)

    r2_values = {
    'Parabel': calculate_adjusted_r_squared(model_parabel, clustered_data_x, clustered_data_y, num_predictors_parabel),
    'Kubisch': calculate_adjusted_r_squared(model_kubisch, clustered_data_x, clustered_data_y, num_predictors_kubisch),
    'Sinus': calculate_adjusted_r_squared(model_sinus, clustered_data_x, clustered_data_y, num_predictors_sinus),
    'Linear': calculate_adjusted_r_squared(model_linear, clustered_data_x, clustered_data_y, num_predictors_linear),
    'Cosinus': calculate_adjusted_r_squared(model_cosinus, clustered_data_x, clustered_data_y, num_predictors_cosinus),
    'Exponential': calculate_adjusted_r_squared(model_exponential, clustered_data_x, clustered_data_y, num_predictors_exponential)
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

    fig, ax = plt.subplots()
    #plt.scatter(x_data, y_data, label='Originaldaten')
    #plt.scatter(df['x_coord'], df['y_coord'], label='Originaldaten')
    plt.scatter(clustered_data_x, clustered_data_y, label='clustering', color='red')


    x_range = np.linspace(clustered_data_x.min(), clustered_data_x.max(), 100).reshape(-1, 1)
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

    plt.xlabel('Magnetization')
    plt.ylabel('Wall Thickness')
    plt.legend()
    plt.show()
"""

def calculate_frequency(x, y):
    # Anwendung der Fast Fourier Transformation auf die Daten
    y_fft = rfft(y)
    x_fft = rfftfreq(n=x.size, d=(x.max()-x.min())/x.size)
    
    # Finden der Frequenz mit der maximalen Amplitude im Spektrum
    idx = np.argmax(np.abs(y_fft))
    frequency = x_fft[idx]
    return frequency

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

    combined_array = np.column_stack((x_data, y_data))
    """
    n_clusters = 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_array)
    cluster_labels = kmeans.labels_

    # Berechne die Distanzen zu den Clusterzentren
    centroids = kmeans.cluster_centers_
    distances = pairwise_distances(combined_array, centroids[cluster_labels], metric='euclidean')
    distances = np.min(distances, axis=1)  # Wähle die minimale Distanz für jeden Punkt

    # Bestimme den Schwellenwert anhand eines Prozentsatzes
    percentile = 90  # Zum Beispiel die 95. Perzentile
    threshold = np.percentile(distances, percentile)
    
    # Ausgabe des Schwellenwerts
    print(f"Schwellenwert für die {percentile}. Perzentile: {threshold}")

    # Optional: Identifiziere und entferne Ausreißer
    inliers = distances < threshold
    """

    dbscan = DBSCAN(eps=2, min_samples=5).fit(combined_array)

    # Erhalte die Cluster-Labels
    cluster_labels = dbscan.labels_

    # Identifiziere die Kernpunkte
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Ausgabe der Anzahl der gefundenen Cluster (ohne Rauschen berücksichtigt)
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f'Geschätzte Anzahl der Cluster: {n_clusters_}')

    # Optional: Identifiziere und entferne Ausreißer
    # Punkte mit dem Label -1 sind Ausreißer
    inliers = cluster_labels != -1
    
    clustered_data_x = np.array(x_data[inliers])
    clustered_data_y = np.array(y_data[inliers])
    mask = ~np.isnan(clustered_data_x) & ~np.isnan(clustered_data_y) & ~np.isinf(clustered_data_x) & ~np.isinf(clustered_data_y)
    
    cleaned_data_x = clustered_data_x[mask]
    cleaned_data_y = clustered_data_y[mask]
    periods = np.diff(np.where(np.diff(np.signbit(cleaned_data_y)))[0])
    calculated_frequency = calculate_frequency(cleaned_data_x, cleaned_data_y)
    initial_guess_sinus = [np.ptp(cleaned_data_y) / 2, calculated_frequency, 0, np.mean(cleaned_data_y)]
    params_sinus, params_covariance_sinus = curve_fit(sinus, cleaned_data_x, cleaned_data_y, p0=initial_guess_sinus, maxfev=10000)
    B = params_sinus[1]
    exclude_zone = 0.01  # Bereich, der um Null ausgeschlossen werden soll
    if -exclude_zone < B < exclude_zone:
        B = exclude_zone if B >= 0 else -exclude_zone
        params_sinus[1] = B
    x_values_for_plot = np.linspace(cleaned_data_x.min(), cleaned_data_x.max(), 500)
    predicted_values_sinus = sinus(x_values_for_plot, *params_sinus)


    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    popt_parabel, _ = curve_fit(parabel, clustered_data_x, clustered_data_y, bounds=bounds, maxfev=10000)
    popt_kubisch, _ = curve_fit(kubisch, clustered_data_x, clustered_data_y)
    
    x0, x1 = clustered_data_x[0], clustered_data_x[-1]
    y0, y1 = clustered_data_y[0], clustered_data_y[-1]
    a_initial = (y1 - y0) / (x1 - x0)
    b_initial = y0 - a_initial * x0
    popt_linear, _ = curve_fit(linear, clustered_data_x, clustered_data_y, p0=[a_initial, b_initial])
    bounds = ([-150, -np.inf, -np.inf], [150, np.inf, np.inf])
    popt_exponential, pcov_exponential = curve_fit(exponential, clustered_data_x, clustered_data_y, bounds=bounds, maxfev=10000)
    a = popt_exponential[0]
    if -exclude_zone < a < exclude_zone:
        # Setzen Sie 'a' auf den Rand des ausgeschlossenen Bereichs, abhängig von seinem Vorzeichen
        a = exclude_zone if a >= 0 else -exclude_zone
        popt_exponential[0] = a
    
    a_initial = 0.01  # Ein kleiner Wert für die Krümmung
    x0, x1 = clustered_data_x[0], clustered_data_x[-1]
    y0, y1 = clustered_data_y[0], clustered_data_y[-1]
    b_initial = (y1 - y0) / (x1 - x0)  # Ähnlich einer linearen Regression
    c_initial = np.mean(clustered_data_y)  # Der Durchschnittswert der y-Daten

    # Überprüfen und Anpassen des Wertes von 'a' und 'b', falls notwendig
    a = popt_parabel[0]
    if -exclude_zone < a < exclude_zone:
        a = exclude_zone if a >= 0 else -exclude_zone
        popt_parabel[0] = a
    y_pred_parabel = parabel(clustered_data_x, *popt_parabel)
    a = popt_kubisch[0]
    if -exclude_zone < a < exclude_zone:
        a = exclude_zone if a >= 0 else -exclude_zone
        popt_kubisch[0] = a
    y_pred_kubisch = kubisch(clustered_data_x, *popt_kubisch)
    y_pred_linear = linear(clustered_data_x, *popt_linear)  
    a = popt_exponential[0]
    b = popt_exponential[1]
    if -exclude_zone < a < exclude_zone:
        a = exclude_zone if a >= 0 else -exclude_zone
        popt_exponential[0] = a
    if -exclude_zone < b < exclude_zone:
        b = exclude_zone if b >= 0 else -exclude_zone
        popt_exponential[1] = b
    y_pred_exponential = exponential(clustered_data_x, *popt_exponential)
    
    num_obs = len(clustered_data_y)
    num_predictors_parabel = 2  # Für eine Parabel (quadratisches Modell)
    num_predictors_kubisch = 3  # Für ein kubisches Modell
    num_predictors_sinus = 4    # Für ein Sinusmodell, angenommen 4 Parameter: Amplitude, Frequenz, Phase, Offset
    num_predictors_linear = 1   # Für ein lineares Modell
    num_predictors_exponential = 2  # Für ein exponentielles Modell

    r2_parabel = r2_score(clustered_data_y, y_pred_parabel)
    r2_kubisch = r2_score(clustered_data_y, y_pred_kubisch) 
    r2_sinus = r2_score(cleaned_data_y, sinus(cleaned_data_x, *params_sinus))
    r2_linear = r2_score(clustered_data_y, y_pred_linear)
    r2_exponential = r2_score(clustered_data_y, y_pred_exponential)
    # Speichern der adjustierten R^2-Werte
    adj_r2_parabel = calculate_adjusted_r2(r2_parabel, num_obs, num_predictors_parabel)
    adj_r2_kubisch = calculate_adjusted_r2(r2_kubisch, num_obs, num_predictors_kubisch)
    adj_r2_sinus = calculate_adjusted_r2(r2_sinus, num_obs, num_predictors_sinus)
    adj_r2_linear = calculate_adjusted_r2(r2_linear, num_obs, num_predictors_linear)
    adj_r2_exponential = calculate_adjusted_r2(r2_exponential, num_obs, num_predictors_exponential)

    r2_values = {
        'Parabel': adj_r2_parabel,
        'Kubisch': adj_r2_kubisch,
        'Sinus': adj_r2_sinus,
        'Linear': adj_r2_linear,
        'Exponential': adj_r2_exponential
}
    best_two = sorted(r2_values, key=r2_values.get, reverse=True)[:2]


    fig, ax = plt.subplots()
    plt.scatter(clustered_data_x, clustered_data_y, label='Originaldaten')

    # Plotte nur die besten zwei Modelle und zeige die angepassten R²-Werte und Funktionsgleichungen
    for model in best_two:
        if model == 'Parabel':
            plt.plot(clustered_data_x, parabel(clustered_data_x, *popt_parabel), label='Beste Parabel-Fit', color='red')
            eq = f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}'
            plt.text(0.95, 0.95, f'Parabel: y = {eq}\nAdj. R² = {adj_r2_parabel:.2f}', 
                    ha='right', va='top', transform=ax.transAxes, color='red')
        elif model == 'Kubisch':
            plt.plot(clustered_data_x, kubisch(clustered_data_x, *popt_kubisch), label='Beste Kubisch-Fit', color='orange')
            eq = f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}'
            plt.text(0.95, 0.85, f'Kubisch: y = {eq}\nAdj. R² = {adj_r2_kubisch:.2f}', 
                    ha='right', va='top', transform=ax.transAxes, color='orange')
        elif model == 'Sinus':
            plt.plot(x_values_for_plot, predicted_values_sinus, label='Fitted Sinus Curve', color='green')
            amplitude, frequency, phase, offset = params_sinus
            eq = f'y = {amplitude:.2f} * sin({frequency:.2f} * x + {phase:.2f}) + {offset:.2f}'
            plt.text(0.95, 0.75, f'Sinus: y = {eq}\nAdj. R² = {adj_r2_sinus:.2f}', 
                    ha='right', va='top', transform=ax.transAxes, color='green')
        elif model == 'Linear':
            plt.plot(clustered_data_x, linear(clustered_data_x, *popt_linear), label='Beste Linear-Fit', color='purple')
            eq = f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}'
            plt.text(0.95, 0.55, f'Linear: y = {eq}\n Adj. R² = {adj_r2_linear:.2f}', 
                    ha='right', va='top', transform=ax.transAxes, color='purple')
        elif model == 'Exponential':
            plt.plot(clustered_data_x, exponential(clustered_data_x, *popt_exponential), label='Beste Exponential-Fit', color='magenta')
            eq = f'{popt_exponential[0]:.2f} * exp({popt_exponential[1]:.2f} * x)'
            plt.text(0.95, 0.45, f'Exponential: y = {eq}\nAdj. R² = {adj_r2_exponential:.2f}', 
                    ha='right', va='top', transform=ax.transAxes, color='magenta')
    
    print("Adjusted R-squared for Parabel: ", adj_r2_parabel,"; ", "Parabel :",f'{popt_parabel[0]:.2f} * x^2 + {popt_parabel[1]:.2f} * x + {popt_parabel[2]:.2f}')
    print("Adjusted R-squared for Kubisch: ", adj_r2_kubisch,"; ", "Kubisch :",f'{popt_kubisch[0]:.2f} * x^3 + {popt_kubisch[1]:.2f} * x^2 + {popt_kubisch[2]:.2f} * x + {popt_kubisch[3]:.2f}')
    amplitude, frequency, phase, offset = params_sinus
    print("Adjusted R-squared for Sinus: ", adj_r2_sinus,"; " "Sinus :",f'y = {amplitude:.2f} * sin({frequency:.2f} * x + {phase:.2f}) + {offset:.2f}')
    print("Adjusted R-squared for Linear: ", adj_r2_linear,"; " "Linear :",f'{popt_linear[0]:.2f} * x + {popt_linear[1]:.2f}')
    print("Adjusted R-squared for Exponential: ", adj_r2_exponential,"; " "Exponential:" f'{popt_exponential[0]:.2f} * exp({popt_exponential[1]:.2f} * x)')
    
    plt.xlabel('Magnetization')
    plt.ylabel('Wall Thickness')
    plt.legend()
    plt.show()

