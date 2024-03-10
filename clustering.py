# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
from matplotlib import pyplot as plt
import hdbscan
import numpy as np

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
            magnetization = []
            wall_thickness = []
            velocity = []
            
            if len(property_list)>0: # Überprüfen, ob mindestens eine Eigenschaft ausgewählt wurde
                for data_id in selected_rows: # Für jede ausgewählte Zeile
                    punkte=get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                    plotting[data_id]=punkte # Datenpunkte in ein Dictionary speichern

                if set(property_list) == {'magnetization', 'velocity', 'wall_thickness'}:
                    for data_id in plotting: 
                        # Assuming data structure: [[magnetization], [velocity], [wall_thickness]]
                        magnetization.extend(plotting[data_id][property_list.index('magnetization')])
                        wall_thickness.extend(plotting[data_id][property_list.index('wall_thickness')])
                        velocity.extend(plotting[data_id][property_list.index('velocity')])

                    #Creating a color map based on velocity
                    #Normalizing velocity values for color mapping
                    norm = plt.Normalize(min(velocity), max(velocity))
                    colors = plt.cm.viridis(norm(velocity))
                    
                    # Scatter plot of magnetization vs wall thickness with colors based on velocity
                    plt.scatter(magnetization, wall_thickness, c=colors)
                    plt.xlabel('Magnetization')
                    plt.ylabel('Wall Thickness')
                    try:
                        plt.show(block=True) 
                        plt.close()
                    except:
                        pass  
                else:
                    for data_id in plotting:  # Für jede ausgewählte Zeile
                        # Datenpunkte vorbereiten
                        data = np.array(plotting[data_id])

                        # Clustering durchführen
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
                        cluster_labels = clusterer.fit_predict(data)

                        # Plot vorbereiten
                        plt.figure()
                        unique_labels = set(cluster_labels)
                        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

                        # Jeden Cluster plotten
                        for k, col in zip(unique_labels, colors):
                            if k == -1:
                                # Schwarz für Ausreißer
                                col = [0, 0, 0, 1]

                            class_member_mask = (cluster_labels == k)
                            xy = data[class_member_mask]
                            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

                        plt.title('HDBSCAN clustering')
                        plt.xlabel(property_list[0])
                        plt.ylabel(property_list[1])
                        plt.show()
                    except:
                        pass
            else:
                # Fehlermeldung, wenn keine Eigenschaft ausgewählt wurde
                sg.popup_quick_message("Please select at least one property", auto_close=True, auto_close_duration=2)

        continue # Nächste Benutzerinteraktion lesen
