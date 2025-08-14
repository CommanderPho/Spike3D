import subprocess
import json
import os
import platform

## Exports VSCode version and the versions of all extensions to a .json file

def get_vscode_version():
    result = subprocess.run(['code', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.splitlines()[0] if result.returncode == 0 else None

def get_extensions_versions():
    result = subprocess.run(['code', '--list-extensions', '--show-versions'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {}
    extensions = {}
    for line in result.stdout.splitlines():
        ext, ver = line.split('@')
        extensions[ext] = ver
    return extensions

def export_versions_to_file(file_path):
    vscode_version = get_vscode_version()
    extensions_versions = get_extensions_versions()

    if not vscode_version:
        raise RuntimeError("Failed to retrieve VSCode version.")

    data = {
        'vscode_version': vscode_version,
        'extensions_versions': extensions_versions
    }

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'vscode_version_info.json')
    export_versions_to_file(file_path)
    print(f"VSCode versions and extensions exported to {file_path}")

