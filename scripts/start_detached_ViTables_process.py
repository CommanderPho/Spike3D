import subprocess
import sys



def launch_vitables_external(file_path=r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\pipeline_results.h5'):
    """ 
        from Spike3D.scripts.start_detached_ViTables_process import launch_vitables_external

    """
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    #si.wShowWindow = subprocess.SW_HIDE # default
    # command = [sys.executable, '-m', 'vitables', file_path] # Adjust the file path as needed
    command = [sys.executable, 'scripts/start_detached_ViTables_process.py', file_path] # Adjust the file path as needed

    # Launch the process detached from the parent
    if sys.platform == 'win32':
        subprocess.Popen(command, creationflags=(subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)) #  | subprocess.CREATE_NO_WINDOW #, startupinfo=si) , shell=True # , startupinfo=si
    else:
        print(f'NON win32!')
        subprocess.Popen(command, start_new_session=True)


launch_vitables_external(file_path=r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\output\pipeline_results.h5')