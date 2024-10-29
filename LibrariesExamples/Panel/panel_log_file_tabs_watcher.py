import queue
import threading
import param
import panel as pn
from pathlib import Path
pn.extension('terminal')
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# path_to_watch = r"K:\scratch\gen_scripts" # Set the path to the directory containing log files

path_to_watch = "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs"

# Define file paths to open in tabs on startup
# test_files = [r"K:\scratch\gen_scripts\run_kdiba_gor01_one_2006-6-08_14-26-15\debug_2024-05-28_03-05-55.Apogee.kdiba.gor01.one.2006-6-08_14-26-15.log",
#     r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-08_21-16-25\debug_2024-05-28_03-05-28.Apogee.kdiba.gor01.two.2006-6-08_21-16-25.log",
#     r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-07_16-40-19\debug_2024-05-28_03-05-24.Apogee.kdiba.gor01.two.2006-6-07_16-40-19.log",
#     r"K:\scratch\gen_scripts\run_kdiba_gor01_two_2006-6-09_22-24-40\debug_2024-05-28_04-05-38.Apogee.kdiba.gor01.two.2006-6-09_22-24-40.log",
# ]

# test_files = ['C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/gor01=one=2006-6-08_14-26-15.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/gor01=two=2006-6-07_16-40-19.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/pin01=one=fet11-01_12-58-54.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/gor01=one=2006-6-09_1-22-43.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/gor01=one=2006-6-12_15-55-31.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/gor01=two=2006-6-12_16-53-46.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/vvp01=two=2006-4-10_12-58-3.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/vvp01=one=2006-4-09_17-29-30.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/vvp01=two=2006-4-09_16-40-54.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/vvp01=one=2006-4-10_12-25-50.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/pin01=one=11-03_12-3-25.log']

test_files = ['C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-08_14-26-15.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_two_2006-6-07_16-40-19.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_pin01_one_fet11-01_12-58-54.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-09_1-22-43.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-12_15-55-31.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_two_2006-6-12_16-53-46.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_two_2006-4-10_12-58-3.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_one_2006-4-09_17-29-30.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_two_2006-4-09_16-40-54.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_one_2006-4-10_12-25-50.log',
 'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_pin01_one_11-03_12-3-25.log']


# Thread-safe queue for communication between file watcher and Panel UI
log_queue = queue.Queue()

# Create a parameterized class to hold our terminals and file content
class LogFileTab(param.Parameterized):
    terminal = param.ClassSelector(class_=pn.widgets.Terminal)
    file_content = param.String(default='')
    
    def __init__(self, file_path, **params):
        super().__init__(**params)
        self.file_path = file_path
        self.terminal = pn.widgets.Terminal(name=file_path, height=400, sizing_mode='stretch_width')
        self.update_content_from_file()
    
    def update_content_from_file(self):
        with open(self.file_path, 'r') as f:
            self.file_content = f.read()
        self.update_terminal()

    @param.depends('file_content', watch=True)
    def update_terminal(self):
        self.terminal.clear()
        self.terminal.write(self.file_content)

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, log_file_tabs):
        super().__init__()
        self.log_file_tabs = log_file_tabs

    def on_modified(self, event):
        if not event.is_directory and event.src_path in self.log_file_tabs:
            log_queue.put((event.src_path, 'modified'))

# Start monitoring the directory with the test_files
def start_monitoring(path, log_file_tabs):
    event_handler = LogFileHandler(log_file_tabs)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)  # Recursive set True if needed
    observer.start()
    return observer


# Setup the observer to watch the test files
log_file_tabs = {file_path: LogFileTab(file_path) for file_path in test_files}
observer = start_monitoring(path_to_watch, log_file_tabs)  # Monitor the parent directory

# Function to periodically check the queue for new log data
def check_for_updates():
    while not log_queue.empty():
        file_path, action = log_queue.get()
        if action == 'modified':
            log_file_tabs[file_path].update_content_from_file()

    # Schedule this function to be called again
    pn.state.add_periodic_callback(check_for_updates, period=500, count=None)

# Create the Panel application with a list of tabs
def create_panel_app(log_file_tabs):
    tabs = pn.Tabs(*[(str(Path(tab.terminal.name).resolve().relative_to(path_to_watch)), tab.terminal) for path, tab in log_file_tabs.items()], closable=True, dynamic=True, tabs_location='left')
    _out = pn.Row(tabs)
    return _out

# Create and show the Panel app
app = create_panel_app(log_file_tabs)
check_for_updates()  # Start the periodic callback to check for updates

if __name__.startswith('bk_script'):
    app.servable()
else:
    app.show(threaded=True)