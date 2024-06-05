import os
import sys
import shutil # for _backup_extant_file(...)
import platform
from contextlib import contextmanager
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any


def copy_file(src_path: str, dest_path: str):
    """
        Copy a file from src_path to dest_path, creating any intermediate directories as needed.
        
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import copy_file
        
    """
    if not isinstance(dest_path, Path):
         dest_path = Path(dest_path).resolve()

    # Create intermediate directories if they don't exist
    dest_directory = dest_path.parent
    dest_directory.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy(src_path, str(dest_path))

    return dest_path


def copy_recursive(source_base_path, target_base_path):
    """ 
    Copy a directory tree from one location to another. This differs from shutil.copytree() that it does not
    require the target destination to not exist. This will copy the contents of one directory in to another
    existing directory without complaining.
    It will create directories if needed, but notify they already existed.
    If will overwrite files if they exist, but notify that they already existed.
    :param source_base_path: Directory
    :param target_base_path:
    :return: None
    
    Source: https://gist.github.com/NanoDano/32bb3ba25b2bd5cdf192542660ac4de0
    
    Usage:
    
        from pyphocorehelpers.Filesystem.path_helpers import copy_recursive
    
    """
    if not Path(target_base_path).exists():
        Path(target_base_path).mkdir()    
    if not Path(source_base_path).is_dir() or not Path(target_base_path).is_dir():
        raise Exception("Source and destination directory and not both directories.\nSource: %s\nTarget: %s" % ( source_base_path, target_base_path))
    for item in os.listdir(source_base_path):
        # Directory
        if os.path.isdir(os.path.join(source_base_path, item)):
            # Create destination directory if needed
            new_target_dir = os.path.join(target_base_path, item)
            try:
                os.mkdir(new_target_dir)
            except OSError:
                sys.stderr.write("WARNING: Directory already exists:\t%s\n" % new_target_dir)

            # Recurse
            new_source_dir = os.path.join(source_base_path, item)
            copy_recursive(new_source_dir, new_target_dir)
        # File
        else:
            # Copy file over
            source_name = os.path.join(source_base_path, item)
            target_name = os.path.join(target_base_path, item)
            if Path(target_name).is_file():
                sys.stderr.write("WARNING: Overwriting existing file:\t%s\n" % target_name)
            shutil.copy(source_name, target_name)
        