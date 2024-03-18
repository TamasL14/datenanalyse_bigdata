import PySimpleGUI as sg
import datetime



# Define the layout
def filter_layout(continents, instruments):
    # Filter elements
    date_from_label = sg.Input(key='-D_FROM-', size=(10, 1))
    date_from_input = sg.CalendarButton(button_text="Startdatum",key="-DATE_FROM-", target='-D_FROM-', format='%Y-%m-%d')
    date_to_label = sg.Input(key='-D_TO-', size=(10, 1))
    date_to_input = sg.CalendarButton(button_text="Enddatum",key="-DATE_TO-", target='-D_TO-', format='%Y-%m-%d')
    continent_label = sg.Text("Continent:", size=(10, 1))
    continent_options = ["Alle", *continents]
    continent_dropdown = sg.Combo(continent_options, key="-CONTINENT-", default_value="Alle")
    instrument_label = sg.Text("Instrument:", size=(10, 1))
    instrument_options = ["Alle", *instruments]
    instrument_dropdown = sg.Combo(instrument_options, key="-INSTRUMENT-", default_value="Alle")
    filter_button = sg.Button("Filter"), sg.Button("Cancel")

    sg.theme("Dark Blue 3")
    layout = [
        [date_from_label,date_from_input,  date_to_label, date_to_input],
        [continent_label, continent_dropdown, instrument_label, instrument_dropdown],
        [filter_button]
    ]
    return layout

# Difinition des Filterfunctions
def filter_data(data_in_collection, date_from, date_to, continent, instrument):
    filtered_data = []
    i = 1
    for row in data_in_collection:
        datum=str(row[2])
        datum=datetime.datetime.strptime(datum,'%Y-%m-%d').date()
        # Ensure filter values are strings (handles cases where they might be already converted)
        if date_from is None:
            date_from = "1970-01-01"
            date_from = datetime.datetime.strptime(date_from,'%Y-%m-%d').date()
        else:
            date_from = str(date_from)
            date_from = datetime.datetime.strptime(date_from,'%Y-%m-%d').date()
        if date_to is None:
            date_to = datetime.datetime.now().strftime('%Y-%m-%d') 
            date_to = datetime.datetime.strptime(date_to,'%Y-%m-%d').date()
        else:
            date_to = str(date_to)
            date_to = datetime.datetime.strptime(date_to,'%Y-%m-%d').date()
        # Apply filtering criteria, ensuring clarity and readability
        if (
            (date_from <= datum <= date_to)
            and (continent == "Alle" or continent == row[3])
            and (instrument == "Alle" or instrument == row[4])
        ):
            filtered_data.append(row)
            filtered_data[i-1][0] = i
            i+=1


    return filtered_data

def filter_table(data_in_collection, continent_options, instrument_options):
    window = sg.Window("Filter Table", filter_layout(continent_options, instrument_options), size=(600, 400))
    event, values = window.read()

    while True:

        if event == sg.WIN_CLOSED or event == "Cancel":
            window.close()
            return data_in_collection # Originelle Daten zurückgeben

        if event == "Filter":
            date_from = None if values["-D_FROM-"] == "" else values["-D_FROM-"]
            date_to = None if values["-D_TO-"] == "" else values["-D_TO-"]
            continent = values["-CONTINENT-"]
            instrument = values["-INSTRUMENT-"]
            try:
                filtered_data = filter_data(data_in_collection, date_from, date_to, continent, instrument)
            except (ValueError, TypeError) as e:
                sg.popup_error(f"Error during filtering: {e}")
                continue

            window.close()  # Close the filter window
            return filtered_data  # Return the filtered data to the main program#


def filter_rows_by_conf_instr(selected_rows, conf, instr):
    #print(data_rows)
    filtered_rows = []
    for row in selected_rows:
        if row[3] == conf and row[4] == instr:  # Annahme: Index 3 ist für Konfiguration, Index 4 für Instrument
            filtered_rows.append(row[1])  # Annahme: Index 1 enthält die data_id
    print(filtered_rows)
    return filtered_rows