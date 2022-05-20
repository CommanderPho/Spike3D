import argparse
import time

import marching_cubes
import numpy
from skimage import measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", help="path to the volume in .npy format")
    ap.add_argument("--view", choices=("library", "skimage"), help="view the result")
    args = ap.parse_args()

    volume = numpy.load(args.volume)
    results = {}

    t0 = time.perf_counter()
    verts, norms, faces = marching_cubes.march(volume, 0)
    t1 = time.perf_counter()
    results["library"] = (verts, faces, norms)
    print(f"library: {t1 - t0:.6} seconds")

    t0 = time.perf_counter()
    verts, faces, norms, _vals = measure.marching_cubes(volume)
    t1 = time.perf_counter()
    results["skimage"] = (verts, faces, norms)
    print(f"skimage: {t1 - t0:.6} seconds")

    if args.view:
        view(*results[args.view])


def view(verts, faces, norms=None):
    from PyQt5.QtWidgets import QApplication
    from pyphoplacecellanalysis.External.pyqtgraph.opengl import GLViewWidget, MeshData
    from pyphoplacecellanalysis.External.pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

    app = QApplication([])
    
    mesh = MeshData(verts, faces)
    if norms is not None:
        mesh.vertexNormals()[...] = norms

    item = GLMeshItem(meshdata=mesh, drawEdges=True, shader="normalColor")
    item.scale(*([0.03] * 3))

    view = GLViewWidget(rotationMethod="quaternion")
    view.addItem(item)
    view.show()

    app.exec()


if __name__ == "__main__":
    main()
