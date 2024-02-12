from sklearn import metrics
from sklearn.cluster import DBSCAN
import os
import json
import PySimpleGUI as sg
from database import get_data_property
from matplotlib import pyplot as plt
import numpy as np

def clustering_window():
    layout = [
        [sg.Text("Wähle bitte zwei Eigenschaften für die DBScan Clustering Analyse", justification="center")],
        [sg.Column([[sg.Check('Defect Channel', key='defect_channel'), sg.Check('Distance', key='distance'), sg.Check('Magnetization', key='magnetization'), sg.Check('Timestamp', key='timestamp'), sg.Check('Velocity', key='velocity'), sg.Check('Wall Thickness', key='wall_thickness')]], justification="center")],
        [sg.Text("Wähle bitte die Anzahl der Cluster",visible=False), sg.InputText(key='k', visible=False)],
        [sg.Text("Wähle bitte min_samples",visible=False), sg.InputText(key='min_samples',visible=False)],
        [sg.Column([[sg.Button("Submit", key='-SUBMIT-'), sg.Button("Cancel", key='-CANCEL-')]], justification="center")],
    ]
    return sg.Window("Clustering", layout, size=(600, 400))

def clustering(selected_rows):
    while True:
        window = clustering_window()
        event, values = window.read()
        if event == '-CANCEL-' or event == sg.WIN_CLOSED:
            window.close()
            break
        if event == '-SUBMIT-':
            punkte=[]
            plotting={}
            property_list=[element for element in values if values[element]==True]
            
            if len(property_list)>0:
                for data_id in selected_rows:
                    punkte=get_data_property(data_id, property_list)
                    plotting[data_id]=punkte

                for data_id in plotting:
                    for punkte in plotting[data_id]:
                        plt.plot(punkte,'o')
                    plt.legend(property_list,loc='upper center')
                    try:
                        plt.show(block=True)
                        plt.close()
                    except:
                        pass
        continue
    return plotting, property_list 
    

    """        try:
                k=int(values['k'])
                min_samples=int(values['min_samples'])
            except:
                pass"""

        
        