import PySimpleGUI as sg
import os
from file_upload_ui import file_upload, folder_upload
from data_prep import prep_file, prep_folder

# Define the window layout
layout = [
    [sg.Text("Willkommen zu ROSEN Clustering")],
    [sg.Button("Upload File"), sg.Button("Upload Folder"), sg.Button("K-Means Clustering"), sg.Button("Cancel")],
]

# Create the window
window = sg.Window("ROSEN Clustering", layout)

# Event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # clicks cancel
    if event == "Cancel" or event == sg.WIN_CLOSED:
        break
    # Process a submitted file
    if event == "Upload File":
        file=file_upload()
        if file==None:
            continue
        elif file.endswith(".h5"):
            prep_file(file)
        else:
            sg.popup_quick_message("Please select a .h5 file")
    if event == "Upload Folder":
        folder=folder_upload()
        if folder==None:
            continue
        elif os.path.isdir(folder):
            prep_folder(folder)
        else:
            sg.popup_quick_message("Please select a folder")
    if event == "Clustering":
        print("Clustering")

window.close()

