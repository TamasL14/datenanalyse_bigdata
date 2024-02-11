from sklearn import metrics
from sklearn.cluster import DBSCAN
import os
import json
import PySimpleGUI as sg

def clustering_window():
    layout = [
        [sg.Text("Wähle bitte zwei Eigenschaften für die DBScan Clustering Analyse", justification="center")],
        [sg.Column([[sg.Check('Defect Channel', key='defect_channel'), sg.Check('Distance', key='distance'), sg.Check('Magnetization', key='magnetization'), sg.Check('Timestamp', key='timestamp'), sg.Check('Velocity', key='velocity'), sg.Check('Wall Thickness', key='wall_thickness')]], justification="center")],
        [sg.Text("Wähle bitte die Anzahl der Cluster"), sg.InputText(key='k')],
        [sg.Text("Wähle bitte min_samples"), sg.InputText(key='min_samples')],
        [sg.Column([[sg.Button("Submit", key='-SUBMIT-'), sg.Button("Cancel", key='-CANCEL-')]], justification="center")]
    ]
    return sg.Window("Clustering", layout, size=(600, 400))



def clustering():
    window = clustering_window()
    event, values = window.read()
    if event == '-CANCEL-' or event == sg.WIN_CLOSED:
        window.close()
        return None
    if event == '-SUBMIT-':
        property_list=[element for element in values if values[element]==True]
        print(property_list)
        try:
            k=int(values['k'])
            min_samples=int(values['min_samples'])
        except:
            pass

        
        