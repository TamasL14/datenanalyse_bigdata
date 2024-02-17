# Hauptdatei des Programms
# In dieser Datei wird das Hauptfenster des Programms erstellt und die Benutzerinteraktionen verarbeitet
# Import Bibliotheken
import PySimpleGUI as sg
import os
import time

# Import Funktionen aus anderen Dateien
from file_upload_ui import file_upload, folder_upload
from data_prep import prep_file, prep_folder
from database import connect_to_db, get_data
from clustering import clustering


# Definition der Hauptlayout
def define_layout(data_in_collection, table_row_colors):
    # Oberste Zeile der Tabelle
    toprow = ['Nr.','id']
    # Tabelle
    table_layout = [[sg.Table(values=data_in_collection, headings=toprow,justification='center', display_row_numbers=False, num_rows=10, key='-DATA_TABLE-', expand_x=True, expand_y=True, enable_events=True, row_colors=table_row_colors, select_mode='browse' )]]
    
    # Fensterlayout: Title, Tabellenlayout, Buttons
    layout = [
        [sg.Text("Willkommen zu ROSEN Clustering", justification='center', size=(40, 1), font=("Helvetica", 25))],
        [sg.Column(table_layout, justification="center",  expand_x=True, expand_y=True)],
        [sg.Column([[sg.Button("Upload File"), sg.Button("Upload Folder"), sg.Button("Plot"), sg.Button("Cancel")]], justification="center")],
    ]
    return layout

#Globale Variablen
loop=0
row_colors = []
selected_rows=[]

# Start des Programms
while True:
    # Wenn das Programm zum ersten Mal gestartet wird oder neugestartet wird
    if loop==0 :

        # Verbindung zur Datenbank
        try:
            connect_to_db()

            # Versuche die Daten aus der Datenbank zu holen
            try:
                # Variablen zurücksetzen
                row_colors = []
                selected_rows=[]

                # Daten über die Funktion get_data() aus der Datei database.py aus der Datenbank holen
                data_in_collection = get_data()

                # Daten in der Tabelle einfügen
                for i in range(len(data_in_collection)):
                    # Zeilennummer und Daten
                    data_in_collection[i] = [i+1, data_in_collection[i]]
                    # Farben der Zeilen
                    row_colors.append([i, 'black','white'])
                
                # Layout definieren
                layout = define_layout(data_in_collection, row_colors)
                # Fenster erstellen
                window = sg.Window("ROSEN Clustering", layout, size=(600, 400),modal=True)
            except:
                # Fehlermeldung, wenn die Daten nicht aus der Datenbank geholt werden können
                sg.popup_quick_message("Data retrieval failed", auto_close=True, auto_close_duration=2)
                time.sleep(2)
                break
        except:
            # Fehlermeldung, wenn die Verbindung zur Datenbank fehlschlägt
            sg.popup_quick_message("Database connection failed", auto_close=True, auto_close_duration=2)
            time.sleep(2)
            break

    # Eingaben des Benutzers lesen
    event, values = window.read()
    
    # Wenn eine Zeile in der Tabelle ausgewählt wird
    if event == '-DATA_TABLE-':
        # ID der ausgewählten Zeile merken
        selected=values['-DATA_TABLE-'][0]
        # Inhalt der ausgewählten Zeile merken
        content_selected=data_in_collection[selected][1]
        # Farbe der ausgewählten Zeile ändern
        if  row_colors[selected][2]=='white':
            # Wenn die Zeile ausgewählt wird, wird die Farbe  der Zeile auf grün gesetzt
            row_colors[selected][2]='green'
            # Inhalt der Zeile in die Liste der ausgewählten Zeilen einfügen
            selected_rows.append(content_selected)
            window['-DATA_TABLE-'].update(row_colors=[[selected, 'black','green']])
        else:
            # Wenn die Zeile abgewählt wird, wird die Farbe wieder auf weiß gesetzt
            row_colors[selected][2]='white'
            # Inhalt der Zeile aus der Liste der ausgewählten Zeilen entfernen
            selected_rows.remove(content_selected)
            window['-DATA_TABLE-'].update(row_colors=[[selected, 'black','white']])
    
    # Wenn der Benutzer das Fenster schließt oder auf "Cancel" klickt, wird das Programm beendet
    if event == "Cancel" or event == sg.WIN_CLOSED:
        window.close()
        break

    # Fenster für das Hochladen einer Datei öffnen
    if event == "Datei hochladen":
        # Fenster schließen und loop zurücksetzen
        window.close()
        loop=0
        # Funktion file_upload() aus der Datei file_upload_ui.py aufrufen
        file=file_upload()
        
        # Wenn keine Datei ausgewählt wird, wird das Programm fortgesetzt
        if file==None:
            continue
        # Prüfen, ob die ausgewählte Datei eine .h5 Datei ist 
        elif file.endswith(".h5"):
            # Die Funktion prep_file() aus der Datei data_prep.py wird aufgerufen
            prep_file(file)
            continue
        else:
            # Fehlermeldung, wenn die ausgewählte Datei keine .h5 Datei ist
            sg.popup_quick_message("Wähle bitte eine .h5 Datei aus!", auto_close=True, auto_close_duration=2)
            time.sleep(1)
            continue
    
    # Fenster für das Hochladen eines Ordners öffnen
    if event == "Ordner hochladen":
        # Fenster schließen und loop zurücksetzen
        window.close()
        loop=0
        # Funktion folder_upload() aus der Datei file_upload_ui.py aufrufen
        folder=folder_upload()
        
        # Wenn kein Ordner ausgewählt wird, wird das Programm fortgesetzt
        if folder==None:
            continue
        # Prüfen, ob der ausgewählte Ordner ein Ordner ist
        elif os.path.isdir(folder):
            # Die Funktion prep_folder() aus der Datei data_prep.py wird aufgerufen
            prep_folder(folder)
            continue
        else:
            # Fehlermeldung, wenn der ausgewählte Ordner kein Ordner ist
            sg.popup_quick_message("Wähle bitte einen Ordner aus!", auto_close=True, auto_close_duration=2)
            time.sleep(1)
            continue

    # Wenn der Benutzer auf "Plot" klickt, wird die Funktion clustering() aus der Datei clustering.py aufgerufen
    if event == "Plot":
        # Fenster schließen und loop zurücksetzen
        window.close()
        loop=0
        # Prüfen, ob mindestens eine Datei aus der Tabelle ausgewählt wurde
        if len(selected_rows)>0:
            try:
                # Die Funktion clustering() aus der Datei clustering.py wird aufgerufen
                clustering(selected_rows)
            except:
                pass
            continue
        else:
            # Fehlermeldung, wenn keine Datei ausgewählt wurde
            sg.popup_quick_message("Wähle bitte mindestens eine Datei aus!", auto_close=True, auto_close_duration=2)
            time.sleep(1)
            continue
    # Loop erhöhen
    loop+=1

# Fenster schließen, wenn das noch nicht geschehen ist
try:
    window.close()
except:
    pass