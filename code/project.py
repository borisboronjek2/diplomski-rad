import sys
import os
os.environ['QT_API'] = 'pyqt6'
import sqlite3
import numpy as np
from scipy.signal import convolve, butter, sosfilt
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialog, QDialogButtonBox, QSlider
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

class MaterialDialog(QDialog):
    def __init__(self, parent=None, material=None):
        super().__init__(parent)
        self.setWindowTitle("Dodaj/Uredi materijal")
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        form_layout.addRow("Naziv:", self.name_edit)

        self.alpha_edits = []
        freqs = ['63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz']
        for freq in freqs:
            edit = QDoubleSpinBox()
            edit.setRange(0.0, 1.0)
            edit.setSingleStep(0.01)
            edit.setValue(0.1)
            self.alpha_edits.append(edit)
            form_layout.addRow(freq + ":", edit)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if material:
            self.name_edit.setText(material[1])
            for i, val in enumerate(material[2:]):
                self.alpha_edits[i].setValue(val)

    def get_data(self):
        name = self.name_edit.text()
        alphas = [edit.value() for edit in self.alpha_edits]
        return name, alphas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Akustička aplikacija za impulsni odziv")
        self.setGeometry(100, 100, 1200, 800)

        # Audio player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.temp_file = None

        # Audio settings
        self.fs = 44100  # Sample rate

        # Database
        self.conn = sqlite3.connect('materials.db')
        self.create_db()

        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Inputs
        self.input_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Ulazi")
        self.setup_input_tab()
        self.update_position_ranges()

        # Tab 2: Materials
        self.materials_tab = QWidget()
        self.tabs.addTab(self.materials_tab, "Materijali")
        self.setup_materials_tab()

        # Tab 3: Visualization
        self.viz_tab = QWidget()
        self.tabs.addTab(self.viz_tab, "Vizualizacija")
        self.setup_viz_tab()

        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Tab 4: Audio
        self.audio_tab = QWidget()
        self.tabs.addTab(self.audio_tab, "Audio")
        self.setup_audio_tab()

        # Tab 5: Results
        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Rezultati")
        self.setup_results_tab()

    def create_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                alpha_63 REAL,
                alpha_125 REAL,
                alpha_250 REAL,
                alpha_500 REAL,
                alpha_1000 REAL,
                alpha_2000 REAL,
                alpha_4000 REAL,
                alpha_8000 REAL
            )
        ''')
        default_materials = [
            ('Beton', 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07),
            ('Drvo', 0.08, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07),
            ('Tepih', 0.1, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3),
            ('Staklo', 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        ]
        for mat in default_materials:
            try:
                cursor.execute('INSERT INTO materials (name, alpha_63, alpha_125, alpha_250, alpha_500, alpha_1000, alpha_2000, alpha_4000, alpha_8000) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', mat)
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()

    def setup_input_tab(self):
        layout = QVBoxLayout(self.input_tab)

        # Room dimensions
        dim_group = QGroupBox("Dimenzije prostorije (m)")
        dim_layout = QFormLayout(dim_group)
        self.length_edit = QDoubleSpinBox()
        self.length_edit.setRange(0.1, 100)
        self.length_edit.setSingleStep(0.1)
        self.length_edit.setValue(25.0)
        dim_layout.addRow("Dužina:", self.length_edit)
        self.width_edit = QDoubleSpinBox()
        self.width_edit.setRange(0.1, 100)
        self.width_edit.setSingleStep(0.1)
        self.width_edit.setValue(14.0)
        dim_layout.addRow("Širina:", self.width_edit)
        self.height_edit = QDoubleSpinBox()
        self.height_edit.setRange(0.1, 100)
        self.height_edit.setSingleStep(0.1)
        self.height_edit.setValue(10.0)
        dim_layout.addRow("Visina:", self.height_edit)
        layout.addWidget(dim_group)

        self.length_edit.valueChanged.connect(self.update_position_ranges)
        self.width_edit.valueChanged.connect(self.update_position_ranges)
        self.height_edit.valueChanged.connect(self.update_position_ranges)

        self.length_edit.valueChanged.connect(self.update_visualization)
        self.width_edit.valueChanged.connect(self.update_visualization)
        self.height_edit.valueChanged.connect(self.update_visualization)

        # Source position
        source_group = QGroupBox("Pozicija izvora zvuka (m)")
        source_layout = QFormLayout(source_group)
        self.source_x = QDoubleSpinBox()
        self.source_x.setRange(0, 100)
        self.source_x.setSingleStep(0.1)
        self.source_x.setValue(1.0)
        source_layout.addRow("X:", self.source_x)
        self.source_y = QDoubleSpinBox()
        self.source_y.setRange(0, 100)
        self.source_y.setSingleStep(0.1)
        self.source_y.setValue(2.0)
        source_layout.addRow("Y:", self.source_y)
        self.source_z = QDoubleSpinBox()
        self.source_z.setRange(0, 100)
        self.source_z.setSingleStep(0.1)
        self.source_z.setValue(1.5)
        source_layout.addRow("Z:", self.source_z)
        layout.addWidget(source_group)

        self.source_x.valueChanged.connect(self.update_visualization)
        self.source_y.valueChanged.connect(self.update_visualization)
        self.source_z.valueChanged.connect(self.update_visualization)

        # Listener position
        listener_group = QGroupBox("Pozicija slušatelja (m)")
        listener_layout = QFormLayout(listener_group)
        self.listener_x = QDoubleSpinBox()
        self.listener_x.setRange(0, 100)
        self.listener_x.setSingleStep(0.1)
        self.listener_x.setValue(3.0)
        listener_layout.addRow("X:", self.listener_x)
        self.listener_y = QDoubleSpinBox()
        self.listener_y.setRange(0, 100)
        self.listener_y.setSingleStep(0.1)
        self.listener_y.setValue(2.0)
        listener_layout.addRow("Y:", self.listener_y)
        self.listener_z = QDoubleSpinBox()
        self.listener_z.setRange(0, 100)
        self.listener_z.setSingleStep(0.1)
        self.listener_z.setValue(1.5)
        listener_layout.addRow("Z:", self.listener_z)
        layout.addWidget(listener_group)

        # Update visualization when listener position changes
        self.listener_x.valueChanged.connect(self.update_visualization)
        self.listener_y.valueChanged.connect(self.update_visualization)
        self.listener_z.valueChanged.connect(self.update_visualization)

        # Materials
        mat_group = QGroupBox("Materijali ploha")
        mat_layout = QFormLayout(mat_group)
        self.floor_mat = QComboBox()
        self.ceiling_mat = QComboBox()
        self.wall1_mat = QComboBox()
        self.wall2_mat = QComboBox()
        self.wall3_mat = QComboBox()
        self.wall4_mat = QComboBox()
        self.load_materials()
        mat_layout.addRow("Pod:", self.floor_mat)
        mat_layout.addRow("Strop:", self.ceiling_mat)
        mat_layout.addRow("Zid X=0:", self.wall1_mat)
        mat_layout.addRow("Zid X=L:", self.wall2_mat)
        mat_layout.addRow("Zid Y=0:", self.wall3_mat)
        mat_layout.addRow("Zid Y=W:", self.wall4_mat)
        layout.addWidget(mat_group)

        # Calculate button
        self.calc_button = QPushButton("Izračunaj impulsni odziv")
        self.calc_button.clicked.connect(self.calculate_ir)
        layout.addWidget(self.calc_button)

        layout.addStretch()

    def update_position_ranges(self):
        L = self.length_edit.value()
        W = self.width_edit.value()
        H = self.height_edit.value()
        self.source_x.setRange(0, L)
        self.source_y.setRange(0, W)
        self.source_z.setRange(0, H)
        self.listener_x.setRange(0, L)
        self.listener_y.setRange(0, W)
        self.listener_z.setRange(0, H)

    def load_materials(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT name FROM materials')
        materials = [row[0] for row in cursor.fetchall()]
        for combo in [self.floor_mat, self.ceiling_mat, self.wall1_mat, self.wall2_mat, self.wall3_mat, self.wall4_mat]:
            combo.clear()
            combo.addItems(materials)

    def get_material_alpha(self, name):
        cursor = self.conn.cursor()
        cursor.execute('SELECT alpha_63, alpha_125, alpha_250, alpha_500, alpha_1000, alpha_2000, alpha_4000, alpha_8000 FROM materials WHERE name = ?', (name,))
        row = cursor.fetchone()
        if row:
            return np.array(row, dtype=float)
        return np.ones(8, dtype=float) * 0.1

    def get_surface_coefficients(self):
        mats = [self.floor_mat.currentText(), self.ceiling_mat.currentText(),
                self.wall1_mat.currentText(), self.wall2_mat.currentText(),
                self.wall3_mat.currentText(), self.wall4_mat.currentText()]
        alpha_arrays = [self.get_material_alpha(mat) for mat in mats]
        reflection_coefs = [np.sqrt(np.clip(1 - alpha.mean(), 0.0, 1.0)) for alpha in alpha_arrays]
        return alpha_arrays, reflection_coefs

    def reflection_path_coefficient(self, i, j, k, r_coefs):
        coeff = 1.0
        for count in range(abs(i)):
            if i > 0:
                wall = 3 if count % 2 == 0 else 2
            else:
                wall = 2 if count % 2 == 0 else 3
            coeff *= r_coefs[wall]
        for count in range(abs(j)):
            if j > 0:
                wall = 5 if count % 2 == 0 else 4
            else:
                wall = 4 if count % 2 == 0 else 5
            coeff *= r_coefs[wall]
        for count in range(abs(k)):
            if k > 0:
                wall = 1 if count % 2 == 0 else 0
            else:
                wall = 0 if count % 2 == 0 else 1
            coeff *= r_coefs[wall]
        return coeff

    def setup_materials_tab(self):
        layout = QVBoxLayout(self.materials_tab)

        self.materials_table = QTableWidget()
        self.materials_table.setColumnCount(9)
        self.materials_table.setHorizontalHeaderLabels(['Naziv', '63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz'])
        layout.addWidget(self.materials_table)

        buttons_layout = QHBoxLayout()
        add_button = QPushButton("Dodaj")
        add_button.clicked.connect(self.add_material)
        buttons_layout.addWidget(add_button)
        edit_button = QPushButton("Uredi")
        edit_button.clicked.connect(self.edit_material)
        buttons_layout.addWidget(edit_button)
        delete_button = QPushButton("Obriši")
        delete_button.clicked.connect(self.delete_material)
        buttons_layout.addWidget(delete_button)
        refresh_button = QPushButton("Reset na default")
        refresh_button.clicked.connect(self.reset_to_default)
        buttons_layout.addWidget(refresh_button)
        layout.addLayout(buttons_layout)

        self.load_materials_table()

    def load_materials_table(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT name, alpha_63, alpha_125, alpha_250, alpha_500, alpha_1000, alpha_2000, alpha_4000, alpha_8000 FROM materials')
        rows = cursor.fetchall()
        self.materials_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                if j == 0:
                    text = str(val)
                else:
                    text = f"{val:.2f}"
                self.materials_table.setItem(i, j, QTableWidgetItem(text))

    def add_material(self):
        dialog = MaterialDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, alphas = dialog.get_data()
            if name:
                cursor = self.conn.cursor()
                try:
                    cursor.execute('INSERT INTO materials (name, alpha_63, alpha_125, alpha_250, alpha_500, alpha_1000, alpha_2000, alpha_4000, alpha_8000) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                   (name, *alphas))
                    self.conn.commit()
                    self.load_materials_table()
                    self.load_materials()
                except sqlite3.IntegrityError:
                    QMessageBox.warning(self, "Upozorenje", "Materijal s tim nazivom već postoji")

    def edit_material(self):
        current_row = self.materials_table.currentRow()
        if current_row >= 0:
            name = self.materials_table.item(current_row, 0).text()
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM materials WHERE name = ?', (name,))
            material = cursor.fetchone()
            if material:
                dialog = MaterialDialog(self, material)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    new_name, alphas = dialog.get_data()
                    if new_name:
                        try:
                            cursor.execute('UPDATE materials SET name = ?, alpha_63 = ?, alpha_125 = ?, alpha_250 = ?, alpha_500 = ?, alpha_1000 = ?, alpha_2000 = ?, alpha_4000 = ?, alpha_8000 = ? WHERE name = ?',
                                           (new_name, *alphas, name))
                            self.conn.commit()
                            self.load_materials_table()
                            self.load_materials()
                        except sqlite3.IntegrityError:
                            QMessageBox.warning(self, "Upozorenje", "Materijal s tim nazivom već postoji")

    def delete_material(self):
        current_row = self.materials_table.currentRow()
        if current_row >= 0:
            name = self.materials_table.item(current_row, 0).text()
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM materials WHERE name = ?', (name,))
            self.conn.commit()
            self.load_materials_table()
            self.load_materials()

    def reset_to_default(self):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM materials')
        default_materials = [
            ('Beton', 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07),
            ('Drvo', 0.08, 0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07),
            ('Tepih', 0.1, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3),
            ('Staklo', 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
        ]
        for mat in default_materials:
            cursor.execute('INSERT INTO materials (name, alpha_63, alpha_125, alpha_250, alpha_500, alpha_1000, alpha_2000, alpha_4000, alpha_8000) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', mat)
        self.conn.commit()
        self.load_materials_table()
        self.load_materials()

    def setup_viz_tab(self):
        layout = QVBoxLayout(self.viz_tab)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        reflection_layout = QHBoxLayout()
        reflection_layout.addWidget(QLabel("Razina refleksije:"))
        self.reflection_level = QSpinBox()
        self.reflection_level.setRange(0, 3)
        self.reflection_level.setValue(0)
        self.reflection_level.valueChanged.connect(self.update_visualization)
        reflection_layout.addWidget(self.reflection_level)
        layout.addLayout(reflection_layout)

        viz_button = QPushButton("Vrati prikaz")
        viz_button.clicked.connect(self.visualize_room)
        layout.addWidget(viz_button)

    def update_visualization(self):
        if self.tabs.currentWidget() == self.viz_tab:
            self.visualize_room()

    def on_tab_changed(self, index):
        if self.tabs.widget(index) == self.viz_tab:
            self.visualize_room()

    def visualize_room(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        L = self.length_edit.value()
        W = self.width_edit.value()
        H = self.height_edit.value()
        # Draw room
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [0, 0, 0, 0, 0], 'k-')
        ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], [H, H, H, H, H], 'k-')
        for i in range(4):
            ax.plot([i*L/3, i*L/3], [0, W], [0, 0], 'k--')
            ax.plot([i*L/3, i*L/3], [0, W], [H, H], 'k--')
            ax.plot([0, L], [i*W/3, i*W/3], [0, 0], 'k--')
            ax.plot([0, L], [i*W/3, i*W/3], [H, H], 'k--')
        # Source
        sx, sy, sz = self.source_x.value(), self.source_y.value(), self.source_z.value()
        ax.scatter(sx, sy, sz, c='r', marker='o', s=100, label='Izvor')
        # Listener
        lx, ly, lz = self.listener_x.value(), self.listener_y.value(), self.listener_z.value()
        ax.scatter(lx, ly, lz, c='b', marker='^', s=100, label='Slušatelj')

        # Virtual sources using image-source method
        n = self.reflection_level.value()
        if n > 0:
            virtual_sources_by_order = {1: [], 2: [], 3: []}
            colors_by_order = {1: [], 2: [], 3: []}
            
            # Generate all virtual sources up to order n
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    for k in range(-n, n+1):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        order = abs(i) + abs(j) + abs(k)
                        if order <= n:
                            if i % 2 == 0:
                                vx = i * L + sx
                            else:
                                vx = (i + 1) * L - sx

                            if j % 2 == 0:
                                vy = j * W + sy
                            else:
                                vy = (j + 1) * W - sy

                            if k % 2 == 0:
                                vz = k * H + sz
                            else:
                                vz = (k + 1) * H - sz
                            virtual_sources_by_order[order].append((vx, vy, vz))
            
            # Plot sources by order with different colors
            color_map = {1: 'red', 2: 'blue', 3: 'green'}
            for order in range(1, n+1):
                if virtual_sources_by_order[order]:
                    sources = virtual_sources_by_order[order]
                    ax.scatter([v[0] for v in sources], [v[1] for v in sources], [v[2] for v in sources],
                              c=color_map[order], marker='x', alpha=0.8, s=60, 
                              label=f'{order}. red refleksije ({len(sources)} izvora)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Vizualizacija prostorije s virtualnim izvorima')
        ax.legend()
        ax.set_box_aspect((L, W, H))
        self.canvas.draw()

    def setup_audio_tab(self):
        layout = QVBoxLayout(self.audio_tab)

        load_button = QPushButton("Učitaj .wav datoteku")
        load_button.clicked.connect(self.load_wav)
        layout.addWidget(load_button)

        self.wav_label = QLabel("Nema učitane datoteke")
        layout.addWidget(self.wav_label)

        # Label for currently playing audio
        self.playing_label = QLabel("Nema reprodukcije")
        layout.addWidget(self.playing_label)

        # Slider for position
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)

        # Connect slider
        self.slider.sliderMoved.connect(self.set_position)
        self.media_player.positionChanged.connect(self.update_slider)
        self.media_player.durationChanged.connect(self.update_slider_range)

        auralize_button = QPushButton("Auraliziraj")
        auralize_button.clicked.connect(self.auralize)
        layout.addWidget(auralize_button)

        self.save_auralized_button = QPushButton("Spremi auralizirani zvuk")
        self.save_auralized_button.setEnabled(False)
        self.save_auralized_button.clicked.connect(self.save_auralized)
        layout.addWidget(self.save_auralized_button)

        # Play controls
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_audio)
        controls_layout.addWidget(self.play_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_audio)
        controls_layout.addWidget(self.pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        controls_layout.addWidget(self.stop_button)
        layout.addLayout(controls_layout)

        layout.addStretch()

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        self.ir_figure = plt.figure()
        self.ir_canvas = FigureCanvas(self.ir_figure)
        layout.addWidget(self.ir_canvas)

        rt_layout = QVBoxLayout()
        self.rt60_label = QLabel("Vrijeme odjeka (Sabine): N/A")
        rt_layout.addWidget(self.rt60_label)
        self.eyring_label = QLabel("Vrijeme odjeka (Eyring): N/A")
        rt_layout.addWidget(self.eyring_label)
        layout.addLayout(rt_layout)

        self.rt60_table = QTableWidget()
        self.rt60_table.setColumnCount(3)
        self.rt60_table.setHorizontalHeaderLabels(['Pojas', 'Sabine (s)', 'Eyring (s)'])
        self.rt60_table.setRowCount(8)
        bands = ['63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz']
        for i, band in enumerate(bands):
            self.rt60_table.setItem(i, 0, QTableWidgetItem(band))
            self.rt60_table.setItem(i, 1, QTableWidgetItem('N/A'))
            self.rt60_table.setItem(i, 2, QTableWidgetItem('N/A'))
        layout.addWidget(self.rt60_table)

    def calculate_ir(self):
        c = 343.0
        L = self.length_edit.value()
        W = self.width_edit.value()
        H = self.height_edit.value()
        sx, sy, sz = self.source_x.value(), self.source_y.value(), self.source_z.value()
        lx, ly, lz = self.listener_x.value(), self.listener_y.value(), self.listener_z.value()

        mats = [self.floor_mat.currentText(), self.ceiling_mat.currentText(),
                self.wall1_mat.currentText(), self.wall2_mat.currentText(),
                self.wall3_mat.currentText(), self.wall4_mat.currentText()]
        alpha_arrays, r_coefs = self.get_surface_coefficients()
        alpha_values = [np.mean(alpha) for alpha in alpha_arrays]

        max_dist = np.sqrt((3 * L)**2 + (3 * W)**2 + (3 * H)**2)
        max_delay = max_dist / c + 0.5
        ir_length = int(max_delay * self.fs)
        self.ir = np.zeros(ir_length)
        self.direct_ir = np.zeros(ir_length)
        self.reflection_ir = np.zeros(ir_length)

        # Direct sound
        direct_dist = np.sqrt((sx - lx)**2 + (sy - ly)**2 + (sz - lz)**2)
        direct_delay = int(np.round(direct_dist / c * self.fs))
        direct_amplitude = 1.0 / (direct_dist + 0.1)
        if direct_delay < ir_length:
            self.direct_ir[direct_delay] += direct_amplitude
            self.ir[direct_delay] += direct_amplitude
        self.direct_dist = direct_dist

        max_order = 1
        max_reflection_delay = 0
        for i in range(-max_order, max_order + 1):
            for j in range(-max_order, max_order + 1):
                for k in range(-max_order, max_order + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    order = abs(i) + abs(j) + abs(k)
                    if order > max_order:
                        continue

                    if i % 2 == 0:
                        vx = i * L + sx
                    else:
                        vx = (i + 1) * L - sx

                    if j % 2 == 0:
                        vy = j * W + sy
                    else:
                        vy = (j + 1) * W - sy

                    if k % 2 == 0:
                        vz = k * H + sz
                    else:
                        vz = (k + 1) * H - sz
                    dist = np.sqrt((vx - lx)**2 + (vy - ly)**2 + (vz - lz)**2)
                    delay = int(np.round(dist / c * self.fs))
                    if delay >= ir_length:
                        continue

                    path_coef = self.reflection_path_coefficient(i, j, k, r_coefs)
                    correction = (path_coef * direct_dist / (dist + 0.1))**2
                    amplitude = direct_amplitude * correction
                    self.direct_ir[delay] += amplitude
                    self.ir[delay] += amplitude
                    max_reflection_delay = max(max_reflection_delay, delay)

        V = L * W * H
        areas = [L * W, L * W, W * H, W * H, L * H, L * H]

        total_abs_sabine = np.zeros(8)
        total_abs_eyring = np.zeros(8)
        for alpha_band, area in zip(alpha_arrays, areas):
            total_abs_sabine += alpha_band * area
            total_abs_eyring += (-np.log(np.clip(1 - alpha_band, 1e-6, 1.0))) * area

        rt60_sabine = np.zeros(8)
        rt60_eyring = np.zeros(8)
        for idx in range(8):
            if total_abs_sabine[idx] > 0:
                rt60_sabine[idx] = 0.161 * V / total_abs_sabine[idx]
            if total_abs_eyring[idx] > 0:
                rt60_eyring[idx] = 0.161 * V / total_abs_eyring[idx]

        avg_rt60_sabine = np.mean(rt60_sabine)
        avg_rt60_eyring = np.mean(rt60_eyring)
        self.rt60_label.setText(f"Vrijeme odjeka (Sabine): {avg_rt60_sabine:.2f} s")
        self.eyring_label.setText(f"Vrijeme odjeka (Eyring): {avg_rt60_eyring:.2f} s")

        for idx in range(8):
            self.rt60_table.setItem(idx, 1, QTableWidgetItem(f"{rt60_sabine[idx]:.2f}"))
            self.rt60_table.setItem(idx, 2, QTableWidgetItem(f"{rt60_eyring[idx]:.2f}"))

        rt60_for_tail = avg_rt60_eyring if avg_rt60_eyring > 0 else avg_rt60_sabine
        if rt60_for_tail <= 0:
            rt60_for_tail = 1.0

        predelay = 2.0 * V ** (1.0 / 3.0) / 1000.0
        tail_start = int(direct_delay + predelay * self.fs)

        if tail_start < ir_length:
            tail_len = ir_length - tail_start

            window_size = int(0.02 * self.fs)
            search_start = max(0, tail_start - window_size)
            if search_start < tail_start:
                reflections_segment = np.abs(self.reflection_ir[search_start:tail_start])
                if reflections_segment.size and np.any(reflections_segment > 0):
                    start_amp = np.mean(reflections_segment[reflections_segment > 0])
                else:
                    start_amp = np.max(np.abs(self.ir[search_start:tail_start]))
                start_amp = max(start_amp, 0.02)
            else:
                start_amp = 0.02

            bands = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
            rt60_bands = rt60_sabine.copy()
            max_rt60 = np.max(rt60_bands[np.isfinite(rt60_bands)]) if np.any(np.isfinite(rt60_bands)) else 1.0
            if max_rt60 <= 0:
                max_rt60 = 1.0

            noise_len_sec = min(30.0, 1.5 * max_rt60)
            noise_len = max(1, int(noise_len_sec * self.fs))
            white_noise = np.random.normal(0.0, 1.0, noise_len)

            final_tail = np.zeros(tail_len)
            t_tail = np.arange(tail_len) / self.fs

            for b_idx, center in enumerate(bands):
                low = center / np.sqrt(2.0)
                high = center * np.sqrt(2.0)
                nyq = self.fs / 2.0
                if low >= nyq:
                    band_signal = np.zeros(tail_len)
                    continue
                if high >= nyq:
                    high = nyq * 0.999

                try:
                    sos = butter(4, [low, high], btype='band', fs=self.fs, output='sos')
                    band_noise = sosfilt(sos, white_noise)
                except Exception:
                    band_noise = np.zeros_like(white_noise)

                rt60_band = rt60_bands[b_idx] if rt60_bands[b_idx] > 0 else max_rt60
                env_len = len(band_noise)
                env_t = np.arange(env_len) / self.fs
                envelope = np.exp(-13.82 * env_t / rt60_band)
                band_tail = band_noise * envelope

                if env_len < tail_len:
                    repeats = int(np.ceil(tail_len / env_len))
                    band_tail_long = np.tile(band_tail, repeats)[:tail_len]
                else:
                    band_tail_long = band_tail[:tail_len]

                final_tail += band_tail_long

            peak = np.max(np.abs(final_tail))
            if peak > 0:
                final_tail = final_tail / peak * start_amp
            else:
                tail_time = t_tail
                final_tail = start_amp * np.exp(-13.82 * tail_time / (rt60_for_tail if rt60_for_tail > 0 else 1.0))

            self.ir[tail_start:tail_start + tail_len] += final_tail
            self.reflection_ir[tail_start:tail_start + tail_len] += final_tail

        self.rt60_sabine = rt60_sabine
        self.predelay = 2.0 * V ** (1.0 / 3.0) / 1000.0

        self.plot_ir()

    def load_wav(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Odaberi .wav datoteku", "", "WAV files (*.wav)")
        if file_name:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            self.wav_label.setText(f"Učitana: {file_name}")
            self.slider.setEnabled(True)
            self.wav_data, self.samplerate = sf.read(file_name)
            self.original_wav = self.wav_data.copy()
            self.current_filename = os.path.basename(file_name)
            self.playing_label.setText(self.current_filename)
            self.save_auralized_button.setEnabled(False)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def update_slider(self, position):
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(position)
            self.slider.blockSignals(False)

    def update_slider_range(self, duration):
        self.slider.setRange(0, duration)

    def play_audio(self):
        self.media_player.play()

    def pause_audio(self):
        self.media_player.pause()

    def stop_audio(self):
        self.media_player.stop()

    def plot_ir(self):
        self.ir_figure.clear()
        ax = self.ir_figure.add_subplot(111)
        time_axis = np.arange(len(self.ir)) / self.fs
        
        display_ir = np.abs(self.ir)
        ax.plot(time_axis, display_ir, label='Odjek')
        if hasattr(self, 'direct_ir'):
            ax.plot(time_axis, self.direct_ir, color='red', linewidth=1.5, label='Direktni zvuk i rane refleksije')
        ax.set_title('Impulsni odziv')
        ax.set_xlabel('Vrijeme (s)')
        ax.set_ylabel('Amplituda')
        ax.legend()
        self.ir_canvas.draw()

    def auralize(self):
        if not hasattr(self, 'ir'):
            self.calculate_ir()
        if hasattr(self, 'wav_data') and hasattr(self, 'ir'):
            self.media_player.stop()

            if self.wav_data.ndim == 1:
                input_audio = self.wav_data.copy()
            else:
                input_audio = np.mean(self.wav_data, axis=1)

            bands = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
            rt60_bands = self.rt60_sabine if hasattr(self, 'rt60_sabine') else np.ones(8)
            max_rt60 = np.max(rt60_bands[np.isfinite(rt60_bands)]) if np.any(np.isfinite(rt60_bands)) else 1.0
            if max_rt60 <= 0:
                max_rt60 = 1.0

            noise_len_sec = min(30.0, 1.5 * max_rt60)
            noise_len = max(1, int(noise_len_sec * self.fs))
            white_noise = np.random.normal(0.0, 1.0, noise_len)

            predelay_sec = self.predelay if hasattr(self, 'predelay') else 0.1
            predelay_samples = int(predelay_sec * self.fs)

            final_auralized = np.zeros(len(input_audio) + len(self.ir))
            nyq = self.fs / 2.0

            for b_idx, center in enumerate(bands):
                low = center / np.sqrt(2.0)
                high = center * np.sqrt(2.0)

                if low >= nyq:
                    continue
                if high >= nyq:
                    high = nyq * 0.999

                try:
                    sos = butter(4, [low, high], btype='band', fs=self.fs, output='sos')
                    band_input = sosfilt(sos, input_audio)
                    band_noise = sosfilt(sos, white_noise)
                except Exception:
                    band_input = np.zeros_like(input_audio)
                    band_noise = np.zeros_like(white_noise)

                rt60_band = rt60_bands[b_idx] if rt60_bands[b_idx] > 0 else max_rt60
                env_t = np.arange(len(band_noise)) / self.fs
                envelope = np.exp(-13.82 * env_t / rt60_band)
                band_tail_impulse = band_noise * envelope

                if predelay_samples > 0:
                    band_tail_impulse = np.concatenate((np.zeros(predelay_samples, dtype=band_tail_impulse.dtype), band_tail_impulse))

                peak_tail = np.max(np.abs(band_tail_impulse))
                peak_refl = np.max(np.abs(self.reflection_ir))
                direct_dist = getattr(self, 'direct_dist', 1.0)
                distance_factor = max(0.2, min(direct_dist / 3.0, 1.0))
                tail_gain = 0.15 * (1.0 + min(direct_dist, 6.0) / 3.0)
                tail_gain = min(tail_gain, 0.4)
                if peak_tail > 0 and peak_refl > 0:
                    band_tail_impulse = band_tail_impulse / peak_tail * peak_refl * tail_gain

                early_ir = 2.0 * self.direct_ir
                band_convolved = convolve(band_input, early_ir, mode='full')
                band_convolved_tail = convolve(band_input, band_tail_impulse, mode='full')
                
                max_len = max(len(band_convolved), len(band_convolved_tail))
                if len(band_convolved) < max_len:
                    band_convolved = np.pad(band_convolved, (0, max_len - len(band_convolved)), mode='constant')
                if len(band_convolved_tail) < max_len:
                    band_convolved_tail = np.pad(band_convolved_tail, (0, max_len - len(band_convolved_tail)), mode='constant')
                
                band_convolved += 0.2 * band_convolved_tail * distance_factor

                if len(final_auralized) < len(band_convolved):
                    final_auralized = np.pad(final_auralized, (0, len(band_convolved) - len(final_auralized)), mode='constant')
                
                final_auralized[:len(band_convolved)] += band_convolved

            max_val = np.max(np.abs(final_auralized))
            if max_val > 0:
                final_auralized = final_auralized / max_val

            self.auralized = final_auralized.astype(np.float32)

            if self.temp_file:
                try:
                    if os.path.exists(self.temp_file.name):
                        os.unlink(self.temp_file.name)
                except:
                    pass

            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(self.temp_file.name, self.auralized, self.samplerate)
            self.media_player.setSource(QUrl.fromLocalFile(self.temp_file.name))
            self.playing_label.setText(f"{self.current_filename} (auralizirano)")
            self.save_auralized_button.setEnabled(True)
            QMessageBox.information(self, "Info", "Auralizacija završena")
        else:
            QMessageBox.warning(self, "Upozorenje", "Učitajte wav i izračunajte IR prvo")

    def save_auralized(self):
        if not hasattr(self, 'auralized'):
            QMessageBox.warning(self, "Upozorenje", "Nema auraliziranog zvuka za spremanje")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Spremi auralizirani zvuk", "auralized.wav", "WAV files (*.wav)")
        if file_name:
            try:
                sf.write(file_name, self.auralized, self.samplerate)
                QMessageBox.information(self, "Info", f"Auralizirani zvuk je spremljen u {file_name}")
            except Exception as e:
                QMessageBox.warning(self, "Greška", f"Neuspjelo spremanje zvuka: {e}")

    def closeEvent(self, event):
        self.media_player.stop()
        self.conn.close()
        if self.temp_file:
            try:
                if os.path.exists(self.temp_file.name):
                    os.unlink(self.temp_file.name)
            except:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())