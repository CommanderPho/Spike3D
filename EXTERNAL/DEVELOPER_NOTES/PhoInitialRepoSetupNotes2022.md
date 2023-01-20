
# Create a new environment named 'phoviz_new' from scratch:
mamba create -n phoviz_new python=3.9
conda activate phoviz_new

mamba install jupyterlab ipykernel pyscaffold pyscaffoldext-dsproject tox -c conda-forge

mamba install matplotlib pyqtconsole numpy pandas pyqtgraph scipy cupy numba jupyter-rfb colorcet -c conda-forge

mamba install seaborn pymc pyvista vaex ipympl hdf5storage ipywidgets ipygany ipyvolume ipyvtklink panel


mamba install jupyter_bokeh bqplot -c conda-forge


pip install PySide2 PyOpenGL PyOpenGL_accelerate




# New environment:

mamba env create -f environment.yml
conda activate demo-pyphocorehelpers



## New 2022-03-19 'phoviz' environment:

conda create -n phoviz python=3.9 ipython ipykernel pyscaffold pyscaffoldext-dsproject tox matplotlib numpy pandas pyqtgraph scipy cupy numba jupyter-rfb hdf5storage ipympl panel bqplot jupyter_bokeh pyside2 -c conda-forge



mamba install ipython ipykernel pyscaffold pyscaffoldext-dsproject tox matplotlib numpy pandas pyqtgraph scipy cupy numba jupyter-rfb hdf5storage ipympl panel bqplot jupyter_bokeh pyside2 -c conda-forge

mamba install ipython ipykernel pyscaffold pyscaffoldext-dsproject tox matplotlib numpy pandas pyqtgraph scipy cupy numba jupyter-rfb hdf5storage ipympl panel bqplot jupyter_bokeh -c conda-forge

## New 2022-06-09 Rachel Setup:

git clone -b pho_variant --single-branch https://github.com/CommanderPho/NeuroPy.git
git clone -b develop --single-branch https://github.com/CommanderPho/pyPhoCoreHelpers.git
git clone -b develop --single-branch https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git
git clone -b develop --single-branch https://github.com/CommanderPho/Spike3D.git


## New 2022-06-29 PyInstaller with Venv setup:
python -m pip install ipython ipykernel numpy pandas scipy cupy numba hdf5storage pingouin indexed PyQt5Singleton

# Visualization:
matplotlib seaborn pymc pyvista vaex ipympl jupyter-rfb colorcet PyOpenGL PyOpenGL_accelerate ipywidgets ipygany ipyvolume ipyvtklink panel jupyter_bokeh bqplot

# Dev-only:
pyscaffold pyscaffoldext-dsproject tox 



# Final Full Command:
python -m pip install ipython ipykernel numpy pandas scipy cupy numba hdf5storage indexed pybursts PyQt5Singleton pingouin matplotlib 



python -m pip install matplotlib seaborn pymc pyqt5 pyvista pyvistaqt vaex ipympl jupyter-rfb colorcet PyOpenGL PyOpenGL_accelerate ipywidgets ipygany ipyvolume ipyvtklink panel jupyter_bokeh bqplot qtawesome vedo 