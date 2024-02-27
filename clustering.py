# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

# Definition des Clustering Fensters
def clustering_window():
    layout = [
        [sg.Text("Wähle bitte zwei Eigenschaften für die HDBSCAN Clustering Analyse", justification="center")],
        [sg.Column([[sg.Check('Defect Channel', key='defect_channel'), sg.Check('Distance', key='distance'), sg.Check('Magnetization', key='magnetization'), sg.Check('Timestamp', key='timestamp'), sg.Check('Velocity', key='velocity'), sg.Check('Wall Thickness', key='wall_thickness')]], justification="center")],
        [sg.Text("Wähle bitte min_samples"), sg.InputText(key='min_samples')],
        [sg.Text("Wähle bitte min_cluster_size"), sg.InputText(key='min_cluster_size')],
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
            property_list=[element for element in values if values[element]==True] # Ausgewählte Eigenschaften in einer Liste speichern
            min_samples = int(values['min_samples'])
            min_cluster_size = int(values['min_cluster_size'])

            if len(property_list)>1: # Überprüfen, ob mindestens zwei Eigenschaften ausgewählt wurden
                # Datenpunkte für die ausgewählten Eigenschaften aus der Datenbank holen
                data_points = []
                for data_id in selected_rows:
                    data_points.append(get_data_property(data_id, property_list))
                data_points = np.array(data_points)

                # Daten skalieren und dimensionale Reduktion durchführen
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_points.reshape(-1, data_points.shape[-1])).reshape(data_points.shape)
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(scaled_data.reshape(-1, scaled_data.shape[-1])).reshape(scaled_data.shape[0], -1)

                # Clustering mit HDBSCAN durchführen
                clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
                labels = clusterer.fit_predict(reduced_data)

                # Farbpalette für die Cluster generieren
                palette = np.array(sns.color_palette("husl", len(set(labels))))

                # Plot erstellen
                plt.figure(figsize=(8, 6))
                for cluster in set(labels):
                    cluster_points = reduced_data[labels == cluster]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=palette[cluster], label=f'Cluster {cluster}' if cluster != -1 else 'Noise', alpha=0.7, s=30)
                plt.title('HDBSCAN Clustering')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend()
                plt.show()

            else:
                # Fehlermeldung, wenn nicht genügend Eigenschaften ausgewählt wurden
                sg.popup_quick_message("Please select at least two properties", auto_close=True, auto_close_duration=2)
        
        continue # Nächste Benutzerinteraktion lesen
