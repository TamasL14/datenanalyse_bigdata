# Dieses Modul öffnet das Fenster für die Clustering Analyse
# Import Bibliotheken
import PySimpleGUI as sg
from database import get_data_property
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from filter_table import filter_rows_by_conf_instr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def polynom(x, a, b, c):
    return a * x**2 + b * x + c

def sinus(x, a, b, c):
    return a * np.sin(b * x + c)

def exponential(x, a, b):
    return a * np.exp(b * x)

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
                    """
                    x_data = df['wall_thickness']
                    y_data = df['magnetization']
                    
                    popt_poly, _ = curve_fit(polynom, x_data, y_data)
                    popt_sinus, _ = curve_fit(sinus, x_data, y_data)
                    popt_exp, _ = curve_fit(exponential, x_data, y_data)

                    y_poly_fit = polynom(x_data, *popt_poly)
                    y_sinus_fit = sinus(x_data, *popt_sinus)
                    y_exp_fit = exponential(x_data, *popt_exp)
                    """


                    xy = np.vstack([df['wall_thickness'], df['magnetization']])
                    kde = gaussian_kde(xy)(xy)
                    norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                    sizes = (1 - norm_velocity) * 50

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
                    """plt.plot(x_data, y_poly_fit, label='Polynom-Fit', color='red')
                    plt.plot(x_data, y_sinus_fit, label='Sinus-Fit', color='green')
                    plt.plot(x_data, y_exp_fit, label='Exponential-Fit', color='blue')
                    """
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

# Standard plotall
def plotall(selected_rows):
    property_list = ['magnetization', 'velocity', 'wall_thickness'] 
    #configuration = ['Africa', 'Europe', 'Australia', 'Africa', 'America']
    #instruments = ['Dolphin', 'Pufferfish', 'Unicorn', 'Dog', 'Elephant']
    instruments = ['Dolphin']
    configuration = ['Africa']
    # Anzahl der Subplots bestimmen
    num_configs = len(configuration)
    num_instrs = len(instruments)
    fig, axs = plt.subplots(num_configs, num_instrs, figsize=(5*num_instrs, 5*num_configs))

    for i, conf in enumerate(configuration):
        for j, instr in enumerate(instruments):
            # Daten vorbereiten
            df = pd.DataFrame(columns=['data_id', 'magnetization', 'velocity', 'wall_thickness'])
            plotting = {}
                       
            relevant_rows = filter_rows_by_conf_instr(selected_rows, conf, instr)
            print(relevant_rows)
            for data_id in relevant_rows: # Für jede relevante Zeile
                punkte = get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                plotting[data_id] = punkte
            
            for data_id in selected_rows: # Für jede ausgewählte Zeile
                punkte = get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                plotting[data_id] = punkte # Datenpunkte in ein Dictionary speichern
            
            for data_id in plotting:
                daten = plotting[data_id]
                temp_df = pd.DataFrame(daten).transpose()
                temp_df.columns = property_list
                temp_df['data_id'] = data_id
                df = pd.concat([df, temp_df], ignore_index=True)

            # Plot für aktuelle Kombination erstellen
            ax = axs[i, j] if num_configs > 1 and num_instrs > 1 else axs[max(i, j)]
            xy = np.vstack([df['wall_thickness'], df['magnetization']])
            
            kde = gaussian_kde(xy)(xy)
            norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
            sizes = (1 - norm_velocity) * 50

            scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
            fig.colorbar(scatter, ax=ax, label='Density')
            ax.set_ylabel('Magnetization')
            ax.set_xlabel('Wall Thickness')
            ax.set_title(f'Configuration: {conf}, Instrument: {instr}')

    plt.tight_layout()
    plt.show()

# Hier muss die get_data_property-Funktion definiert werden.
# Beispiel:
# def get_data_property(data_id, property_list):
#     # Code, um Datenpunkte zu erhalten




"""
def plotall(selected_rows):
    while True:

            # Variablen initialisieren
            punkte=[]
            plotting={}
            property_list=['magnetization', 'velocity', 'wall_thickness'] 
            configuration=['Africa', 'Europe', 'Australia', 'Africa', 'America']
            instruments=['Dolphin', 'Pufferfish', 'Unicorn', 'Dog', 'Elephant']


            df = pd.DataFrame(columns=['data_id', 'magnetization', 'velocity', 'wall_thickness'])
            for conf in configuration:
                for instr in instruments:
                    for data_id in selected_rows: # Für jede ausgewählte Zeile
                        punkte=get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                        plotting[data_id]=punkte # Datenpunkte in ein Dictionary speichern
                        for data_id in plotting: 
                            # Assuming data structure: [[magnetization], [velocity], [wall_thickness]]
                            daten = plotting[data_id]
                            temp_df = pd.DataFrame(daten).transpose()
                            temp_df.columns = property_list
                            temp_df['data_id'] = data_id
                            df = pd.concat([df, temp_df], ignore_index=True)


                        xy = np.vstack([df['wall_thickness'], df['magnetization']])
                        kde = gaussian_kde(xy)(xy)
                        norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                        sizes = (1 - norm_velocity) * 50

                        fig, ax = plt.subplots()
                        scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
                        fig.colorbar(scatter, ax=ax, label='Density')
                        plt.ylabel('Magnetization')
                        plt.xlabel('Wall Thickness')    
                        try:
                            plt.show(block=True) 
                            plt.close()
                        except:
                            pass  
            else:
                # Fehlermeldung, wenn keine Eigenschaft ausgewählt wurde
             sg.popup_quick_message("Please select at least one property", auto_close=True, auto_close_duration=2)
        
            continue # Nächste Benutzerinteraktion lesen
    """


"""multithread versuch 1
def fetch_and_prepare_data(data_id, property_list):
    # Führen Sie Ihre Datenabfrage und -verarbeitung hier aus
    # Beispiel:
    punkte = get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
    return data_id, punkte

def plotall(selected_rows):
    property_list = ['magnetization', 'velocity', 'wall_thickness'] 
    configuration = ['Africa', 'Europe', 'Australia', 'Africa', 'America']
    instruments = ['Dolphin', 'Pufferfish', 'Unicorn', 'Dog', 'Elephant']

    num_configs = len(configuration)
    num_instrs = len(instruments)
    fig, axs = plt.subplots(num_configs, num_instrs, figsize=(5*num_instrs, 5*num_configs))

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, conf in enumerate(configuration):
            for j, instr in enumerate(instruments):
                future_to_data = {executor.submit(fetch_and_prepare_data, data_id, property_list): data_id for data_id in selected_rows}

                df = pd.DataFrame(columns=['data_id', 'magnetization', 'velocity', 'wall_thickness'])

                for future in concurrent.futures.as_completed(future_to_data):
                    data_id, punkte = future.result()
                    temp_df = pd.DataFrame([punkte], columns=property_list)
                    temp_df['data_id'] = data_id
                    df = pd.concat([df, temp_df], ignore_index=True)

                # Plot für aktuelle Kombination erstellen
                ax = axs[i, j] if num_configs > 1 and num_instrs > 1 else axs[max(i, j)]
                xy = np.vstack([df['wall_thickness'], df['magnetization']]).T  # Transponieren Sie das Array, sodass jede Zeile einen Punkt darstellt
                kde = gaussian_kde(xy)(xy) 
                norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                sizes = (1 - norm_velocity) * 50

                scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
                fig.colorbar(scatter, ax=ax, label='Density')
                ax.set_ylabel('Magnetization')
                ax.set_xlabel('Wall Thickness')
                ax.set_title(f'Configuration: {conf}, Instrument: {instr}')

    plt.tight_layout()
    plt.show()

# Hier muss die get_data_property-Funktion definiert werden.

"""


"""multithread versuch 2
def plotall(selected_rows):
    property_list = ['magnetization', 'velocity', 'wall_thickness'] 
    configuration = ['Africa', 'Europe', 'Australia', 'Africa', 'America']
    instruments = ['Dolphin', 'Pufferfish', 'Unicorn', 'Dog', 'Elephant']

    # Anzahl der Subplots bestimmen
    num_configs = len(configuration)
    num_instrs = len(instruments)
    fig, axs = plt.subplots(num_configs, num_instrs, figsize=(5*num_instrs, 5*num_configs))
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, conf in enumerate(configuration):
            for j, instr in enumerate(instruments):
                # Daten vorbereiten
                df = pd.DataFrame(columns=['data_id', 'magnetization', 'velocity', 'wall_thickness'])
                plotting = {}

                for data_id in selected_rows: # Für jede ausgewählte Zeile
                    punkte = get_data_property(data_id, property_list) # Datenpunkte aus der Datenbank holen
                    plotting[data_id] = punkte # Datenpunkte in ein Dictionary speichern
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                for data_id in plotting:
                    daten = plotting[data_id]
                    temp_df = pd.DataFrame(daten).transpose()
                    temp_df.columns = property_list
                    temp_df['data_id'] = data_id
                    df = pd.concat([df, temp_df], ignore_index=True)

                # Plot für aktuelle Kombination erstellen
                ax = axs[i, j] if num_configs > 1 and num_instrs > 1 else axs[max(i, j)]
                xy = np.vstack([df['wall_thickness'], df['magnetization']])
                kde = gaussian_kde(xy)(xy)
                norm_velocity = (df['velocity'] - df['velocity'].min()) / (df['velocity'].max() - df['velocity'].min())
                sizes = (1 - norm_velocity) * 50

                scatter = ax.scatter(df['wall_thickness'], df['magnetization'], c=kde, s=sizes, edgecolor='none', cmap='coolwarm')
                fig.colorbar(scatter, ax=ax, label='Density')
                ax.set_ylabel('Magnetization')
                ax.set_xlabel('Wall Thickness')
                ax.set_title(f'Configuration: {conf}, Instrument: {instr}')



    plt.tight_layout()
    plt.show()
"""