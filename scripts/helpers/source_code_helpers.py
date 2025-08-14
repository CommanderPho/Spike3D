import os
import sys
import pathlib
import re
import hashlib # for hashing pyproject.toml files and seeing if they changed

def find_py_files(project_path, exclude_dirs=[]):
    # Find all .py files in the project directory and its subdirectories
    if not isinstance(project_path, pathlib.Path):
        project_path = pathlib.Path(project_path)
    py_files = project_path.glob("**/*.py")
    py_files = [file_path for file_path in py_files] # to list

    excluded_py_files = []
    if exclude_dirs is not None:
        # Find all .py files in the project directory and its subdirectories, excluding the 'my_exclude_dir' directory
        exclude_paths = [project_path.joinpath(a_dir) for a_dir in exclude_dirs]
        for an_exclude_path in exclude_paths:
            excluded_py_files.extend([file_path for file_path in an_exclude_path.glob("**/*.py")])

    included_py_files = [x for x in py_files if x not in excluded_py_files]
    return included_py_files


def replace_text_in_file(file_path, regex_pattern, replacement_string, debug_print=False):
    with open(file_path, 'r') as file:
        file_content = file.read()

    if debug_print:
        print(f"====================== Read from file ({file_path}) ======================:\n{file_content}")
    
    # updated_content = re.sub(regex_pattern, replacement_string, file_content, flags=re.MULTILINE)
    target_replace_strings = re.findall(regex_pattern, file_content, re.MULTILINE)
    assert len(target_replace_strings) == 1
    target_replace_string = target_replace_strings[0]
    if debug_print:
        print(f'Replacing:\n{target_replace_string}')
        print(f"====================== replacing ======================:\n{target_replace_string}\n\n====================== with replacement string ====================== :\n{replacement_string}\n\n")
    updated_content = file_content.replace(target_replace_string, replacement_string, 1)
    if debug_print:
        print(updated_content)

    if debug_print:
        print(f"======================  updated_content ====================== :\n{updated_content}\n\n")
        print(f"====================== saving to {file_path}...")
    with open(file_path, 'w') as file:
        file.write(updated_content)

def insert_text(source_file, insert_text_str:str, output_file, insertion_string:str='<INSERT_HERE>'):
    """Inserts the text from insert_text_str into the source_file at the insertion_string, and saves the result to output_file.

    Args:
        source_file (_type_): _description_
        insert_text_str (str): _description_
        output_file (_type_): _description_
        insertion_string (str, optional): _description_. Defaults to '<INSERT_HERE>'.
    """
    # Load the source text
    with open(source_file, 'r') as f:
        source_text = f.read()

    # Find the insertion point in the source text
    insert_index = source_text.find(insertion_string)

    # Insert the text
    updated_text = source_text[:insert_index] + insert_text_str + source_text[insert_index:]

    # Save the updated text to the output file
    with open(output_file, 'w') as f:
        f.write(updated_text)

def insert_text_from_file(source_file, insert_file, output_file, insertion_string:str='<INSERT_HERE>'):
    """ Wraps insert_text, but loads the insert_text from a file instead of a string. """
    # Load the insert text
    with open(insert_file, 'r') as f:
        insert_text_str = f.read()
    insert_text(source_file, insert_text_str, output_file, insertion_string)

def hash_text_in_file(file_path, ignore_whitespace:bool=True, ignore_line_comments:bool=True, case_insensitive:bool=True):
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Remove all comments from the string by searching for the '#' character and removing everything from that character to the end of the line.
    if ignore_line_comments:
        file_content = '\n'.join(line.split('#')[0] for line in file_content.split('\n'))

    # remove all whitespace characters (space, tab, newline, and so on)
    if ignore_whitespace:
        file_content = ''.join(file_content.split())

    if case_insensitive:
        file_content = file_content.lower()

    return hashlib.sha256(file_content.encode('utf-8')).hexdigest()

def did_file_hash_change(file_path) -> bool:
    """ Returns True if the file's hash value has changed since the last run by reading f'{file_path}.sha256'. Saves the new hash value to f'{file_path}.sha256'"""
    # Define the path to the previous hash value file
    hash_file_path = f'{file_path}.sha256'

    # Calculate the new hash value
    new_hash_value = hash_text_in_file(file_path)    

    # Check if the hash value file exists
    if os.path.exists(hash_file_path):
        # Read the previous hash value from the file
        with open(hash_file_path, 'r') as f:
            old_hash_value = f.read().strip()

        # Compare the new hash value with the previous hash value
        if new_hash_value == old_hash_value:
            print('The file has *NOT* changed since the last run')
            did_file_change = False
        else:
            print('The file *has* changed since the last run')
            did_file_change = True
    else:
        # No previous hash value file exists:
        did_file_change = True

    if did_file_change:
        # Save the new hash value to the file
        with open(hash_file_path, 'w') as f:
            f.write(new_hash_value)

    return did_file_change


