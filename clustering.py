# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
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
                    xy = np.vstack([magnetization, wall_thickness])
                    kde = gaussian_kde(xy)(xy)
                    norm_velocity = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity))
                    sizes = (1 - norm_velocity) * 100                    

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(magnetization, wall_thickness, c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
                    fig.colorbar(scatter, ax=ax, label='Density')
                    plt.xlabel('Magnetization')
                    plt.ylabel('Wall Thickness')
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