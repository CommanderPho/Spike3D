import threading
import time
import matplotlib
# import matplotlib.backends.backend_qt
import matplotlib.backends.backend_qt5
import matplotlib.pyplot as plt

import mpl_qtthread # `poetry add mpl_qtthread`
""" TODO 2023-03-08 06:00: - [ ] Broken on Windows even after install mpl_qtthread
    raise RuntimeError("teleporter not inited")
RuntimeError: teleporter not inited

"""

matplotlib.use("module://mpl_qtthread.backend_agg")

matplotlib.backends.backend_qt5._create_qApp()

mpl_qtthread.monkeypatch_pyplot()

plt.ion()


def background():
    # time.sleep(1)
    fig, ax = plt.subplots()
    (ln,) = ax.plot(range(5))
    for j in range(5):
        print(f"starting to block {j}")
        ln.set_color(f"C{j}")
        ax.set_title(f'cycle {j}')
        fig.canvas.draw_idle()
        time.sleep(5)
    print("Done! please close the window")

threading.Thread(target=background).start()
matplotlib.backends.backend_qt5.qApp.exec()

