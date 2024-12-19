import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from hitDetection import HitDetector
import configparser
import os
import sv_ttk
import numpy as np


class TabletOptimizerGUI:
    def __init__(self, root):
        self.default_color = "#262626"
        self.detector = None
        self.config_file = './config.ini'
        self.home_dir = ''

        self.root = root
        self.root.title("Tablet Optimizer")

        width = 820
        height = 680
        self.root.maxsize(width, height)
        self.root.minsize(width, height)

        # First File Input
        self.replay_path_frame = ttk.Frame(root)
        self.replay_path_frame.grid(row=0, column=0, padx=10, pady=10)

        self.label1 = ttk.Label(self.replay_path_frame, text="Replay File Path (.osr)")
        self.label1.grid(row=0, column=0, padx=10, pady=10)

        self.replay_path = tk.StringVar()
        self.entry1 = ttk.Entry(self.replay_path_frame, textvariable=self.replay_path, width=50)
        self.entry1.grid(row=0, column=1, padx=10, pady=10)

        self.browse1 = ttk.Button(self.replay_path_frame, text="Browse", command=self.load_file1)
        self.browse1.grid(row=0, column=2, padx=0, pady=10)

        # Second File Input
        self.map_path_frame = ttk.Frame(root)
        self.map_path_frame.grid(row=1, column=0, padx=10, pady=10)

        self.label2 = ttk.Label(self.map_path_frame, text="Map File Path (.osu)")
        self.label2.grid(row=0, column=0, padx=10, pady=10)

        self.map_path = tk.StringVar()
        self.entry2 = ttk.Entry(self.map_path_frame, textvariable=self.map_path, width=50)
        self.entry2.grid(row=0, column=1, padx=10, pady=10)

        self.browse2 = ttk.Button(self.map_path_frame, text="Browse", command=self.load_file2)
        self.browse2.grid(row=0, column=2, padx=0, pady=10)

        # Plot Frame
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.figure1 = plt.Figure(figsize=(3, 3), dpi=100)
        self.figure1.set_facecolor(self.default_color)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plot_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
        self.canvas1.draw()

        self.figure2 = plt.Figure(figsize=(3, 3), dpi=100)
        self.figure2.set_facecolor(self.default_color)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        self.canvas2.draw()

        # Utility Buttons
        self.calibrate_size_button = ttk.Button(root, text="Calibrate Area", command=self.calibrate_area)
        self.calibrate_size_button.grid(row=1, column=1, padx=10, pady=10)

        self.load_button = ttk.Button(root, text='Load Replay', command=self.process_replay)
        self.load_button.grid(row=0, column=1, padx=10, pady=10)

        # Console box
        self.console = scrolledtext.ScrolledText(root, state='disabled', height=10, width=80, wrap=tk.WORD)
        self.console.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        self.load_config()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def on_close(self):
        self.save_config()
        self.root.destroy()

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            config.read(self.config_file)
            if 'Paths' in config and 'home_dir' in config['Paths']:
                self.home_dir = config['Paths']['home_dir']
                if not os.path.isdir(self.home_dir):
                    self.log_to_console('Saved home directory is invalid.')
                    self.home_dir = self.prompt_for_home_dir()
                    config['Paths']['home_dir'] = self.home_dir
                    self.save_config()
                else:
                    self.log_to_console(f'Home directory set: {self.home_dir}')

            else:
                self.log_to_console('No home directory found in config. Please select one now.')
                if 'Paths' not in config:
                    config['Paths'] = {}
                config['Paths']['home_dir'] = self.prompt_for_home_dir()
                self.save_config()

            if 'Paths' in config:
                self.replay_path.set(config['Paths'].get('replay_path', ''))
                self.map_path.set(config['Paths'].get('map_path', ''))

            if self.replay_path.get() and self.map_path.get():
                self.detector = HitDetector(self.replay_path.get(), self.map_path.get())

            self.log_to_console("Configuration loaded.")
        else:
            self.log_to_console("No config file found.")
            config['Paths'] = {}
            config['Paths']['home_dir'] = self.prompt_for_home_dir()
            self.save_config()

    def prompt_for_home_dir(self):
        self.home_dir = filedialog.askdirectory(title='Select an osu! home directory')
        if not self.home_dir:
            self.home_dir = os.getcwd()
            self.log_to_console('No directory selected. Defaulting to current directory.')
        else:
            self.log_to_console(f'Home directory set to {self.home_dir}')
        return self.home_dir

    def save_config(self):
        config = configparser.ConfigParser()
        config['Paths'] = {
            'file1': self.replay_path.get(),
            'file2': self.map_path.get(),
            'home_dir': self.home_dir
        }
        with open(self.config_file, 'w') as configfile:
            config.write(configfile)
        self.log_to_console("Configuration saved.")

    def log_to_console(self, message):
        self.console.config(state='normal')
        self.console.insert(tk.END, message + '\n')
        self.console.config(state='disabled')
        self.console.see(tk.END)

    def load_file1(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if file_path.endswith('.osr'):
                self.replay_path.set(file_path)
                self.log_to_console(f"Loaded replay: {file_path}")
            else:
                self.log_to_console('Please load a valid osu replay file (.osr)')

    def load_file2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if file_path.endswith('.osu'):
                self.map_path.set(file_path)
                self.log_to_console(f"Loaded map: {file_path}")
            else:
                self.log_to_console('Please load a valid osu map file (.osu)')

    def plot_graph(self, figure, canvas, plot_func):
        figure.clear()
        fig, ax = plot_func()
        if fig and ax:
            canvas.figure = fig
            canvas.draw()
            self.log_to_console("Figure plotted.")

    def process_replay(self):
        self.figure1.clear()
        self.figure1.set_facecolor(self.default_color)
        self.figure2.clear()
        self.figure2.set_facecolor(self.default_color)
        self.canvas1.draw()
        self.canvas2.draw()

        self.log_to_console('Loading replay and map...')
        self.detector = HitDetector(self.replay_path.get(), self.map_path.get())
        self.log_to_console("Processing Replay...")
        self.detector.process_map_data()
        self.plot_graph(self.figure1, self.canvas1, self.detector.plot_hit_errors)
        self.log_to_console('Processing Successful!')

    def calibrate_area(self):
        if not self.detector:
            self.log_to_console('Please specify a valid osu map and replay.')
        else:
            self.log_to_console("Processing angle and mean size deviation...")
            self.detector.process_size()
            self.detector.process_rotation()
            self.plot_graph(self.figure2, self.canvas2, self.detector.plot_adj_hit_errors)
            self.log_to_console('Processing Successful!')
            self.log_to_console(f'[Suggested Tablet Area Adjustments] tilt: '
                                f'{np.degrees(self.detector.adj_theta): .3f} size: {self.detector.adj_size: 3f}')


if __name__ == "__main__":
    root = tk.Tk()
    sv_ttk.set_theme('dark', root)
    app = TabletOptimizerGUI(root)
    root.mainloop()
