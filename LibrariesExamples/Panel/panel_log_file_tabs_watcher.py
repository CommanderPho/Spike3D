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

# test_files = ['C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-08_14-26-15.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_two_2006-6-07_16-40-19.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_pin01_one_fet11-01_12-58-54.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-09_1-22-43.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_one_2006-6-12_15-55-31.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_gor01_two_2006-6-12_16-53-46.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_two_2006-4-10_12-58-3.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_one_2006-4-09_17-29-30.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_two_2006-4-09_16-40-54.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_vvp01_one_2006-4-10_12-25-50.log',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/data/neptune/logs/merged/kdiba_pin01_one_11-03_12-3-25.log']


class LogFilePanel(param.Parameterized):
    """ 
    from Spike3D.LibrariesExamples.Panel.panel_log_file_tabs_watcher import LogFilePanel
    
    """
    path_to_watch = param.String()
    test_files = param.List()
    log_queue = queue.Queue()
    
    def __init__(self, path_to_watch, test_files, **params):
        super().__init__(**params)
        self.path_to_watch = path_to_watch
        self.test_files = test_files
        self.log_file_tabs = {file_path: self.LogFileTab(file_path) for file_path in self.test_files}
        self.observer = self.start_monitoring(self.path_to_watch, self.log_file_tabs)
        self.check_for_updates()  # Start the periodic callback to check for updates

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
                LogFilePanel.log_queue.put((event.src_path, 'modified'))

    def start_monitoring(self, path, log_file_tabs):
        event_handler = self.LogFileHandler(log_file_tabs)
        observer = Observer()
        observer.schedule(event_handler, path, recursive=True)
        observer.start()
        return observer

    def check_for_updates(self):
        while not self.log_queue.empty():
            file_path, action = self.log_queue.get()
            if action == 'modified':
                self.log_file_tabs[file_path].update_content_from_file()
        pn.state.add_periodic_callback(self.check_for_updates, period=500, count=None)

    def create_panel_app(self):
        tabs = pn.Tabs(*[(str(Path(tab.terminal.name).resolve().relative_to(self.path_to_watch)), tab.terminal) for path, tab in self.log_file_tabs.items()], closable=True, dynamic=True, tabs_location='left')
        return pn.Row(tabs)

    def show(self):
        app = self.create_panel_app()
        app.show(threaded=True)