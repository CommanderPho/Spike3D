# -*- mode: python ; coding: utf-8 -*-
#
from vedo import installdir as vedo_installdir
import os
vedo_installdir = os.path.join(vedo_installdir,'fonts')

block_cipher = None

added_files = [
    (vedo_installdir+'/*', 'vedo/fonts/'),
]

a = Analysis(['python_PipelineRun.py'],
             pathex=[],
             binaries=[],
             hiddenimports=[
                 'vtkmodules',
                 'vtkmodules.all',
                 'vtkmodules.util',
                 'vtkmodules.util.numpy_support',
                 'vtkmodules.qt.QVTKRenderWindowInteractor',
             ],
             datas = added_files,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='stager',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
