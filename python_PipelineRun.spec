# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['python_PipelineRun.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['neuropy', 'pyphocorehelpers', 'pyphoplacecellanalysis', 'pyphoplacecellanalysis.External.pyqtgraph', 'pyphoplacecellanalysis.External.pyqtgraph.*',
        'pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt5', 'pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt5', 'pyphoplacecellanalysis.External.pyqtgraph.GraphicsScene.exportDialogTemplate_pyqt5',
        'pyphoplacecellanalysis.External.pyqtgraph.imageview.ImageViewTemplate_pyqt5', 'pyphoplacecellanalysis.External.pyqtgraph.flowchart.FlowchartCtrlTemplate_pyqt5', 'pyphoplacecellanalysis.External.pyqtgraph.console.template_pyqt5',
        'pyphoplacecellanalysis.External.pyqtgraph.canvas.TransformGuiTemplate_pyqt5', 'pyphoplacecellanalysis.External.pyqtgraph.canvas.CanvasTemplate_pyqt5'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=True,
    win_private_assemblies=True,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Spike3D_PipelineRun',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Spike3D_PipelineRun',
)
