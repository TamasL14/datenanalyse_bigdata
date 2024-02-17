# Dieses Modul öffnet die Fenster für das Hochladen von Dateien und Ordnern
# Import Bibliotheken
import PySimpleGUI as sg

# Definition des File Upload Fensters
def file_upload_window():
    layout = [
        [sg.Text("Please select a file"), sg.InputText(), sg.FileBrowse()],
        [sg.Button("Submit"), sg.Button("Cancel")],
    ]
    return sg.Window("File Browser", layout) # Fenseter erstellen

# Definition des Folder Upload Fensters
def folder_upload_window():
    layout = [
        [sg.Text("Please select a folder"), sg.InputText(), sg.FolderBrowse()],
        [sg.Button("Submit"), sg.Button("Cancel")],
    ]   
    return sg.Window("Folder Browser", layout) # Fenseter erstellen

# Funktion zum Öffnen des File Upload Fensters
def file_upload():
    window = file_upload_window() # Öffnen des File Upload Fensters   
    event, values = window.read() # Benutzerinteraktion mit dem Fenster lesen

    # Wenn das Fenster geschlossen wird oder der "Cancel" Button gedrückt wird
    if event == "Cancel" or event == sg.WIN_CLOSED: 
        window.close() # Fenster schließen
        return None
    
    if event == "Submit": # Wenn der "Submit" Button gedrückt wird       
        window.close() # Fenster schließen
        return(values[0]) # Rückgabe des ausgewählten Pfades

# Funktion zum Öffnen des Folder Upload Fensters
def folder_upload():   
    window = folder_upload_window() # Öffnen des Folder Upload Fensters    
    event, values = window.read() # Benutzerinteraktion mit dem Fenster lesen
    # Wenn das Fenster geschlossen wird oder der "Cancel" Button gedrückt wird
    if event == "Cancel" or event == sg.WIN_CLOSED:        
        window.close() # Fenster schließen
        return None

    # Wenn der "Submit" Button gedrückt wird
    if event == "Submit":
        window.close() # Fenster schließen
        return(values[0]) # Rückgabe des ausgewählten Pfades
