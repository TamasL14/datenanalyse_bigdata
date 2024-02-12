import PySimpleGUI as sg
import os

# Define the window layout
def file_upload_window():
    layout = [
        [sg.Text("Please select a file"), sg.InputText(), sg.FileBrowse()],
        [sg.Button("Submit"), sg.Button("Cancel")],
    ]
    return sg.Window("File Browser", layout)

def file_upload():
    window = file_upload_window()
    event, values = window.read()
    if event == "Cancel" or event == sg.WIN_CLOSED:
        window.close()
        return None
    # Process a submitted file
    if event == "Submit":
        window.close()
        return(values[0])
    
def folder_upload_window():
    layout = [
        [sg.Text("Please select a folder"), sg.InputText(), sg.FolderBrowse()],
        [sg.Button("Submit"), sg.Button("Cancel")],
    ]
    return sg.Window("Folder Browser", layout)

def folder_upload():
    window = folder_upload_window()
    event, values = window.read()
    if event == "Cancel" or event == sg.WIN_CLOSED:
        window.close()
        return None
    # Process a submitted folder
    if event == "Submit":
        window.close()
        return(values[0])
