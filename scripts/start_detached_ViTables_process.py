import subprocess
import sys

command = [sys.executable, '-m', 'vitables', 'path/to/file'] # Adjust the file path as needed

# Launch the process detached from the parent
if sys.platform == 'win32':
    subprocess.Popen(command, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
else:
    subprocess.Popen(command, start_new_session=True)

