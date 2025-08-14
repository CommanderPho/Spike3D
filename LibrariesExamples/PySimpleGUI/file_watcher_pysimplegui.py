import os
import queue
import threading
import PySimpleGUI as sg
from pyphoplacecellanalysis.General.Batch.LogFileWatcher import LogFileHandler, start_monitoring


sg.theme('DarkAmber')   # Add a touch of color

# # All the stuff inside your window.
# layout = [  [sg.Text('Some text on Row 1')],
#             [sg.Text('Enter something on Row 2'), sg.InputText()],
#             [sg.Button('Ok'), sg.Button('Cancel')] ]

# # Create the Window
# window = sg.Window('Window Title', layout)
# # Event Loop to process "events" and get the "values" of the inputs
# while True:
#     event, values = window.read()
#     if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
#         break
#     print('You entered ', values[0])

# window.close()


# The column with the tab group
file_tabs_column = [
    [sg.TabGroup([[]], key='-TABGROUP-', tab_location='lefttop', enable_events=True, expand_x=True, expand_y=True)]
]

# The initial layout with an empty column
layout = [
    [sg.Column(file_tabs_column, element_justification='center', expand_x=True, expand_y=True)]
]

# Create the window
window = sg.Window('LogFileHandler', layout, resizable=True, finalize=True)

# Start the log monitoring in a separate thread
log_queue = queue.Queue()
path_to_watch = r"K:\scratch\gen_scripts" # Set the path to the directory containing log files
observer = start_monitoring(path_to_watch, log_queue)

test_files = [r"K:\scratch\gen_scripts\run_kdiba_gor01_one_2006-6-08_14-26-15\debug_2024-05-28_03-05-55.Apogee.kdiba.gor01.one.2006-6-08_14-26-15.log",
            r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-08_21-16-25\debug_2024-05-28_03-05-28.Apogee.kdiba.gor01.two.2006-6-08_21-16-25.log",
            r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-07_16-40-19\debug_2024-05-28_03-05-24.Apogee.kdiba.gor01.two.2006-6-07_16-40-19.log",
            r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-09_22-24-40\debug_2024-05-28_04-05-38.Apogee.kdiba.gor01.two.2006-6-09_22-24-40.log",
            ]

for a_file in test_files:
    with open(a_file, 'r') as file:
        contents = file.read()
        log_queue.put((a_file, contents))


tabs = {}
tab_group = window['-TABGROUP-']
running = True

# Main event loop
while running:
    event, values = window.read(timeout=100)

    # Check the queue for new log events
    try:
        while True: # Empty the queue of all events
            filepath, contents = log_queue.get_nowait()
            filename = os.path.basename(filepath)

            # If the tab doesn't exist yet, create it
            if filename not in tabs:
                new_tab_layout = [[sg.Text(contents, size=(80, 20), key=f'FILE_CONTENT_{filename}')]]
                tab_group.add_tab(filename, new_tab_layout)
                tabs[filename] = filepath
            else:
                window[f'FILE_CONTENT_{filename}'].update(contents)

    except queue.Empty:
        pass

    if event == sg.WINDOW_CLOSED:
        running = False

# Clean up
observer.stop()
observer.join()
window.close()