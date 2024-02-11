import PySimpleGUI as sg
import os
import time
from file_upload_ui import file_upload, folder_upload
from data_prep import prep_file, prep_folder
from database import connect_to_db, get_data
from clustering import clustering

def define_layout(data_in_collection, table_row_colors):
    # Define the table headings
    toprow = ['Nr.','id']
    # Define Table layout
    table_layout = [[sg.Table(values=data_in_collection, headings=toprow,justification='center', display_row_numbers=False, num_rows=10, key='-DATA_TABLE-', expand_x=True, expand_y=True, enable_events=True, row_colors=table_row_colors, select_mode='browse' )]]
    
    # Define the window layout
    layout = [
        [sg.Text("Willkommen zu ROSEN Clustering", justification='center', size=(40, 1), font=("Helvetica", 25))],
        [sg.Column(table_layout, justification="center",  expand_x=True, expand_y=True)],
        [sg.Column([[sg.Button("Upload File"), sg.Button("Upload Folder"), sg.Button("Clustering"), sg.Button("Cancel")]], justification="center")],
    ]
    return layout
loop=0
row_colors = []
selected_rows=[]
# Event loop
while True:
    
    # Connect to the database
    if loop==0 :
        try:
            connect_to_db()
            # Get the data from the database
            try:
                row_colors = []
                selected_rows=[]
                data_in_collection = get_data()
                for i in range(len(data_in_collection)):
                    data_in_collection[i] = [i+1, data_in_collection[i]]
                    row_colors.append([i, 'black','white'])
                # Define the layout
                layout = define_layout(data_in_collection, row_colors)
                window = sg.Window("ROSEN Clustering", layout, size=(600, 400),modal=True)
            except:
                sg.popup_quick_message("Data retrieval failed", auto_close=True, auto_close_duration=2)
                time.sleep(2)
                break
        except:
            sg.popup_quick_message("Database connection failed", auto_close=True, auto_close_duration=2)
            time.sleep(2)
            break

    # Read the window
    event, values = window.read()
    
    # Process the table event
    if event == '-DATA_TABLE-':  # Table event
        selected=values['-DATA_TABLE-'][0]
        content_selected=data_in_collection[selected][1]
        if  row_colors[selected][2]=='white':
            row_colors[selected][2]='green'
            selected_rows.append(content_selected)
            window['-DATA_TABLE-'].update(row_colors=[[selected, 'black','green']])
        else:
            row_colors[selected][2]='white'
            selected_rows.remove(content_selected)
            window['-DATA_TABLE-'].update(row_colors=[[selected, 'black','white']])
    
    # End program if user closes window or clicks cancel
    if event == "Cancel" or event == sg.WIN_CLOSED:
        window.close()
        break

    # Process a submitted file
    if event == "Upload File":
        window.close()
        file=file_upload()
        loop=0

        if file==None:
            continue
        elif file.endswith(".h5"):
            prep_file(file)
            continue
        else:
            sg.popup_quick_message("Wähle bitte eine .h5 Datei aus", auto_close=True, auto_close_duration=2)
            time.sleep(2)
            continue
    
    # Process a submitted folder
    if event == "Upload Folder":
        window.close()
        folder=folder_upload()
        loop=0

        if folder==None:
            continue
        elif os.path.isdir(folder):
            prep_folder(folder)
            continue
        else:
            sg.popup_quick_message("Wähle bitte einen Ordner aus", auto_close=True, auto_close_duration=2)
            time.sleep(2)
            continue

    #Start clustering
    if event == "Clustering":
        window.close()
        loop=0
        if len(selected_rows)>0:
            asd=clustering()
            
        else:
            sg.popup_quick_message("Wähle bitte mindestens eine Datei aus", auto_close=True, auto_close_duration=2)
            time.sleep(2)
            continue
        if asd==None:
            continue
    loop+=1
# Close the window
try:
    window.close()
except:
    pass