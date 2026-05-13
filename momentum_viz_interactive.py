import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

class ColoredLineItem(pg.GraphicsObject):
    """
    A highly optimized custom pyqtgraph item to draw line segments
    with varying colors (to represent acceleration stress).
    """
    def __init__(self):
        super().__init__()
        self.picture = QtGui.QPicture()
        
    def update_data(self, x, y, colors):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw all segments into a QPicture to be cached for ultra-fast rendering
        for i in range(len(x)-1):
            p.setPen(pg.mkPen(colors[i], width=4, cap=QtCore.Qt.RoundCap))
            p.drawLine(QtCore.QPointF(x[i], y[i]), QtCore.QPointF(x[i+1], y[i+1]))
        p.end()
        
        # Force a redraw
        self.prepareGeometryChange()
        self.update()
        
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
        
    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class KinematicSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinematic Envelope Simulator (PyQtGraph)")
        self.resize(1100, 700)
        
        # Simulation State
        self.dt = 0.02
        self.kp = 25.0
        self.kd = 5.0
        self.playing = True
        self.frame = 0
        self.waypoints = np.array([
            [100, 150], [700, 150], [700, 450],
            [250, 450], [100, 300], [250, 150]
        ], dtype=float)
        
        self.init_ui()
        self.run_simulation()
        
        # Animation Timer (50 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(self.dt * 1000))

    def init_ui(self):
        # Main Widget & Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Left Panel (Controls) ---
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # Title
        title = QtWidgets.QLabel("Kinematic Bounds")
        title.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        left_layout.addWidget(title)
        
        # Play/Pause & Restart Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_restart = QtWidgets.QPushButton("Restart")
        self.btn_restart.clicked.connect(self.restart)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_restart)
        left_layout.addLayout(btn_layout)
        
        # Max Force Slider
        self.lbl_amax = QtWidgets.QLabel("Max Force Limit (a_max): 1200")
        self.slider_amax = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_amax.setRange(200, 5000)
        self.slider_amax.setValue(1200)
        self.slider_amax.valueChanged.connect(self.on_params_changed)
        
        amax_desc = QtWidgets.QLabel("Simulates maximum muscle force. Lower limits force wider, smoother turns.")
        amax_desc.setWordWrap(True)
        amax_desc.setStyleSheet("color: gray; font-size: 11px;")
        
        left_layout.addWidget(self.lbl_amax)
        left_layout.addWidget(amax_desc)
        left_layout.addWidget(self.slider_amax)
        
        # Speed Slider
        self.lbl_speed = QtWidgets.QLabel("Animal Speed (v): 250 px/s")
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(100, 600)
        self.slider_speed.setValue(250)
        self.slider_speed.valueChanged.connect(self.on_params_changed)
        
        speed_desc = QtWidgets.QLabel("Higher momentum requires vastly more force to turn.")
        speed_desc.setWordWrap(True)
        speed_desc.setStyleSheet("color: gray; font-size: 11px;")
        
        left_layout.addWidget(self.lbl_speed)
        left_layout.addWidget(speed_desc)
        left_layout.addWidget(self.slider_speed)
        
        # Live Telemetry Panel
        telemetry_group = QtWidgets.QGroupBox("Live Telemetry")
        telemetry_layout = QtWidgets.QFormLayout(telemetry_group)
        
        self.lbl_val_speed = QtWidgets.QLabel("0")
        self.lbl_val_speed.setFont(QtGui.QFont("Consolas", 12))
        self.lbl_val_accel = QtWidgets.QLabel("0")
        self.lbl_val_accel.setFont(QtGui.QFont("Consolas", 12))
        
        telemetry_layout.addRow("Current Speed:", self.lbl_val_speed)
        telemetry_layout.addRow("Acceleration:", self.lbl_val_accel)
        
        self.force_gauge = QtWidgets.QProgressBar()
        self.force_gauge.setTextVisible(False)
        self.force_gauge.setFixedHeight(10)
        telemetry_layout.addRow(self.force_gauge)
        left_layout.addWidget(telemetry_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- Right Panel (PyQtGraph Plot) ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.getViewBox().invertY(True) # Invert Y to match SVG standard
        self.plot_widget.setXRange(0, 800)
        self.plot_widget.setYRange(50, 550)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.plot_widget)
        
        # Add Plot Items
        # 1. Theoretical dashed path
        self.target_line = pg.PlotCurveItem(pen=pg.mkPen(color=(203, 213, 225), width=2, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.target_line)
        
        # 2. Actual physical path (Colored)
        self.actual_line = ColoredLineItem()
        self.plot_widget.addItem(self.actual_line)
        
        # 3. Target Ghost Dot
        self.dot_target = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(148, 163, 184), pen=None)
        self.plot_widget.addItem(self.dot_target)
        
        # 4. Animal Dot
        self.dot_animal = pg.ScatterPlotItem(size=14, brush=pg.mkBrush(59, 130, 246), pen=None)
        self.plot_widget.addItem(self.dot_animal)

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def restart(self):
        self.frame = 0

    def on_params_changed(self):
        self.lbl_amax.setText(f"Max Force Limit (a_max): {self.slider_amax.value()}")
        self.lbl_speed.setText(f"Animal Speed (v): {self.slider_speed.value()} px/s")
        self.run_simulation()

    def run_simulation(self):
        """Re-calculates the entire trajectory arrays when sliders change."""
        aMax = self.slider_amax.value()
        speed = self.slider_speed.value()
        
        # 1. Generate theoretical target path
        t_path = []
        for i in range(len(self.waypoints)):
            wp = self.waypoints[i]
            next_wp = self.waypoints[(i + 1) % len(self.waypoints)]
            dist = np.hypot(next_wp[0] - wp[0], next_wp[1] - wp[1])
            steps = int(dist / (speed * self.dt))
            
            if steps == 0: continue
            for step in range(steps):
                frac = step / steps
                t_path.append([
                    wp[0] + (next_wp[0] - wp[0]) * frac,
                    wp[1] + (next_wp[1] - wp[1]) * frac
                ])
                
        self.target_path = np.array(t_path)
        
        # 2. Simulate animal following the target
        a_path, v_mags, a_mags, is_maxed = [], [], [], []
        
        pos = np.array(self.target_path[0], dtype=float)
        vel = np.array([0.0, 0.0], dtype=float)
        
        for target in self.target_path:
            # PD Controller attempting to reach the target dot
            ax = self.kp * (target[0] - pos[0]) - self.kd * vel[0]
            ay = self.kp * (target[1] - pos[1]) - self.kd * vel[1]
            
            a_mag = np.hypot(ax, ay)
            actual_a_mag = a_mag
            
            # The Core Physics Constraint
            if a_mag > aMax:
                ax = ax * (aMax / a_mag)
                ay = ay * (aMax / a_mag)
                actual_a_mag = aMax
                
            vel[0] += ax * self.dt
            vel[1] += ay * self.dt
            pos[0] += vel[0] * self.dt
            pos[1] += vel[1] * self.dt
            
            a_path.append(pos.copy())
            v_mags.append(np.hypot(vel[0], vel[1]))
            a_mags.append(actual_a_mag)
            is_maxed.append(actual_a_mag >= aMax - 0.1)
            
        self.actual_path = np.array(a_path)
        self.v_mags = np.array(v_mags)
        self.a_mags = np.array(a_mags)
        self.is_maxed = np.array(is_maxed)
        
        # Update static plot graphics
        self.target_line.setData(self.target_path[:, 0], self.target_path[:, 1])
        
        # Generate colors for the trajectory line (Green -> Red based on stress)
        stress = np.clip(self.a_mags / aMax, 0, 1)
        r = (stress * 255).astype(int)
        g = ((1 - stress) * 200 + 50).astype(int)
        
        colors = [QtGui.QColor(r[i], g[i], 50) for i in range(len(r))]
        self.actual_line.update_data(self.actual_path[:, 0], self.actual_path[:, 1], colors)
        
        # Reset animation safety bound
        if self.frame >= len(self.actual_path):
            self.frame = 0

    def tick(self):
        """Called every frame to update the animation positions and UI"""
        if not self.playing or len(self.actual_path) == 0:
            return
            
        self.frame = (self.frame + 1) % len(self.actual_path)
        
        # Update target dot
        t_pos = self.target_path[self.frame]
        self.dot_target.setData([t_pos[0]], [t_pos[1]])
        
        # Update animal dot & color
        a_pos = self.actual_path[self.frame]
        maxed = self.is_maxed[self.frame]
        brush_color = (239, 68, 68) if maxed else (59, 130, 246) # Red if maxed, Blue otherwise
        self.dot_animal.setData([a_pos[0]], [a_pos[1]], brush=pg.mkBrush(*brush_color))
        
        # Update Telemetry Text
        speed_val = self.v_mags[self.frame]
        accel_val = self.a_mags[self.frame]
        
        self.lbl_val_speed.setText(f"{int(speed_val)}")
        self.lbl_val_accel.setText(f"{int(accel_val)}")
        
        # Color the acceleration text
        if maxed:
            self.lbl_val_accel.setStyleSheet("color: #ef4444; font-weight: bold;")
        else:
            self.lbl_val_accel.setStyleSheet("color: #16a34a;")
            
        # Update Force Gauge
        pct = int(min(100, (accel_val / self.slider_amax.value()) * 100))
        self.force_gauge.setValue(pct)
        if maxed:
            self.force_gauge.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
        else:
            self.force_gauge.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")


if __name__ == '__main__':
    # Enable High DPI scaling for crisp rendering on modern monitors
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    # app = QtWidgets.QApplication(sys.argv)
    app = pg.mkQApp('momentum_viz_sim')
    window = KinematicSimulator()
    window.show()
    sys.exit(app.exec_())