import os
import cv2
import numpy as np
from ftplib import FTP
import shutil
import random
from watchdog.events import FileSystemEventHandler
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys 
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

MIRROR_FILE_PATH = r'mirror_command/mirror_change.txt'
DISPERSION_FILE_PATH = r'dazzler_command/dispersion.txt'

with open(MIRROR_FILE_PATH, 'r') as file:
    content = file.read()
mirror_values = list(map(int, content.split()))

with open(DISPERSION_FILE_PATH, 'r') as file:
    content = file.readlines()

dispersion_values = {
    0: int(content[0].split('=')[1].strip()),  # 0 is the key for 'order2'
}

class BetatronApplication(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super(BetatronApplication, self).__init__(*args, **kwargs)

        self.mean_count_per_n_images  = 0
        self.count_grad = 0
        self.run_count = 0
        self.count_history = np.array([])

        self.epsilon = 1e-8
        self.momentum_decay_one = 0.9
        self.momentum_decay_two = 0.999
        self.initial_focus_learning_rate = 320
        self.initial_second_dispersion_learning_rate = 320

        self.initial_momentum_estimate = 0
        self.initial_squared_gradient = 0 

        # self.initial_momentum_estimate_history = 0
        # self.initial_squared_gradient_history = 0 

        self.focus_learning_rate_history = np.array([])
        self.second_dispersion_learning_rate_history = np.array([])
        self.momentum_estimate_history = np.array([])
        self.squared_gradient_history = np.array([])

        self.biased_momentum_estimate_history = np.array([])
        self.biased_squared_gradient_history = np.array([])

    # ------------ Plotting ------------ #

        self.second_dispersion_der_history = np.array([])
        self.focus_der_history = np.array([])
        self.total_gradient_history = np.array([])

        self.iteration_data = np.array([])
        self.der_iteration_data = np.array([])
        self.count_data = np.array([])
        
        self.count_plot_widget = pg.PlotWidget()
        self.count_plot_widget.setWindowTitle('count optimization')
        self.count_plot_widget.setLabel('left', 'count')
        self.count_plot_widget.setLabel('bottom', 'n_images iteration')
        self.count_plot_widget.showGrid(x=True, y=True)
        self.count_plot_widget.show()

        self.main_plot_window = pg.GraphicsLayoutWidget()
        self.main_plot_window.show()

        layout = self.main_plot_window.addLayout(row=0, col=0)

        self.count_plot_widget = layout.addPlot(title='count vs n_images iteration')
        self.focus_plot = layout.addPlot(title='count_focus_derivative')
        self.second_dispersion_plot = layout.addPlot(title='count_second_dispersion_derivative')
        self.total_gradient_plot = layout.addPlot(title='total_gradient')

        subplots = [self.count_plot_widget, self.focus_plot, self.second_dispersion_plot, self.total_gradient_plot]
        for subplot in subplots:
            subplot.showGrid(x=True, y=True)

        self.plot_curve = self.count_plot_widget.plot(pen='r')
        self.focus_curve = self.focus_plot.plot(pen='r', name='focus derivative')
        self.second_dispersion_curve = self.second_dispersion_plot.plot(pen='g', name='second dispersion derivative')
        self.total_gradient_curve = self.total_gradient_plot.plot(pen='y', name='total gradient')

        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.der_iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.der_iteration_data, self.second_dispersion_der_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

    # ------------ Deformable mirror ------------ #

        # init -150
        self.MIRROR_HOST = "192.168.200.3"
        self.MIRROR_USER = "Utilisateur"
        self.MIRROR_PASSWORD = "alls"    

        self.initial_focus = -240
        self.focus_history = []    
        # self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        # self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        self.FOCUS_LOWER_BOUND = -99999
        self.FOCUS_UPPER_BOUND = +99999

        self.tolerance = 1
        
    # ------------ Dazzler ------------ #

        self.DAZZLER_HOST = "192.168.58.7"
        self.DAZZLER_USER = "fastlite"
        self.DAZZLER_PASSWORD = "fastlite"

        # 36100 initial 
        self.initial_second_dispersion = -240
        self.second_dispersion_history = []
        # self.SECOND_DISPERSION_LOWER_BOUND = max(self.initial_second_dispersion - 500, 30000)
        # self.SECOND_DISPERSION_UPPER_BOUND = min(self.initial_second_dispersion + 500, 40000)

        self.SECOND_DISPERSION_LOWER_BOUND = -99999
        self.SECOND_DISPERSION_UPPER_BOUND = +99999

        self.random_direction = [random.choice([-1, 1]) for _ in range(4)]

    def upload_files(self):
        mirror_ftp = FTP()
        dazzler_ftp = FTP()

        mirror_ftp.connect(host=self.MIRROR_HOST)
        mirror_ftp.login(user=self.MIRROR_USER, passwd=self.MIRROR_PASSWORD)

        dazzler_ftp.connect(host=self.DAZZLER_HOST)
        dazzler_ftp.login(user=self.DAZZLER_USER, passwd=self.DAZZLER_PASSWORD)

        mirror_files = [os.path.basename(MIRROR_FILE_PATH)]
        dazzler_files = [os.path.basename(DISPERSION_FILE_PATH)]

        for mirror_file_name in mirror_files:
            for dazzler_file_name in dazzler_files:
                focus_file_path = MIRROR_FILE_PATH
                dispersion_file_path = DISPERSION_FILE_PATH

                if os.path.isfile(focus_file_path) and os.path.isfile(dispersion_file_path):
                    copy_mirror_IMG_PATH = os.path.join('mirror_command', f'copy_{mirror_file_name}')
                    copy_dazzler_IMG_PATH = os.path.join('dazzler_command', f'copy_{dazzler_file_name}')

                    try:
                        os.makedirs(os.path.dirname(copy_mirror_IMG_PATH))
                        os.makedirs(os.path.dirname(copy_dazzler_IMG_PATH))
                    except OSError:
                        pass

                    shutil.copy(focus_file_path, copy_mirror_IMG_PATH)
                    shutil.copy(dispersion_file_path, copy_dazzler_IMG_PATH)

                    with open(copy_mirror_IMG_PATH, 'rb') as local_file:
                        mirror_ftp.storbinary(f'STOR {mirror_file_name}', local_file)
                        print(f"Uploaded to mirror FTP: {mirror_file_name}")

                    with open(copy_dazzler_IMG_PATH, 'rb') as local_file:
                        dazzler_ftp.storbinary(f'STOR {dazzler_file_name}', local_file)
                        print(f"Uploaded to dazzler FTP: {dazzler_file_name}")

                    os.remove(copy_mirror_IMG_PATH)
                    os.remove(copy_dazzler_IMG_PATH)

    def write_values(self):

        self.new_focus = round(np.clip(self.focus_history[-1], self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND))
        self.new_second_dispersion = round(np.clip(self.second_dispersion_history[-1], self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND))

        mirror_values[0] = self.new_focus
        dispersion_values[0] = self.new_second_dispersion

        with open(MIRROR_FILE_PATH, 'w') as file:
            file.write(' '.join(map(str, mirror_values)))

        with open(DISPERSION_FILE_PATH, 'w') as file:
            file.write(f'order2 = {dispersion_values[0]}\n')

        # self.upload_files() # send files to second computer

        QtCore.QCoreApplication.processEvents()

    def count_function(self):

        x = self.focus_history[-1]
        y = self.second_dispersion_history[-1]

        count_func = (((0.1 * (x + y)))** 2 * np.sin(0.01 * (x + y)))

        self.count_history = np.append(self.count_history, [count_func]) # this is the count for the value
            
    def calc_derivatives(self):
        x = self.focus_history[-1]
        y = self.second_dispersion_history[-1]

        self.count_focus_der = 0.2*(0.1*(x+y))*np.sin(0.01*(x+y))+0.01*(np.cos(0.01*(x+y)))*(0.1*(x+y))**2
        self.count_second_dispersion_der = 0.2*(0.1*(x+y))*np.sin(0.01*(x+y))+0.01*(np.cos(0.01*(x+y)))*(0.1*(x+y))**2

        self.focus_der_history = np.append(self.focus_der_history, [self.count_focus_der])

        self.second_dispersion_der_history = np.append(self.second_dispersion_der_history, [self.count_second_dispersion_der])

        self.total_gradient = (self.focus_der_history[-1] + self.second_dispersion_der_history[-1])

        self.total_gradient_history = np.append(self.total_gradient_history, [self.total_gradient])

        return {"focus":self.count_focus_der,"second_dispersion":self.count_second_dispersion_der}

    def calc_estimated_momentum(self):
        
        # calculate unbiased momentum estimate
        self.new_momentum_estimate = (self.momentum_decay_one*self.momentum_estimate_history[-1]) + ((1-self.momentum_decay_one)*(self.focus_der_history[-1]))
        
        self.momentum_estimate_history = np.append(self.momentum_estimate_history, [self.new_momentum_estimate])
        
        # calculate biased momentum estimate
        self.new_biased_momentum = ((self.momentum_estimate_history[-1])/(1-((self.momentum_decay_one)**self.run_count)))
        
        self.biased_momentum_estimate_history = np.append(self.biased_momentum_estimate_history, [self.new_biased_momentum])
        

    def calc_squared_grad(self):
        
        # calculate unbiased squared gradient
        self.new_squared_gradient_estimate = (self.momentum_decay_two*self.squared_gradient_history[-1]) + ((1-self.momentum_decay_two)*((self.focus_der_history[-1])**2))
        
        self.squared_gradient_history = np.append(self.squared_gradient_history, [self.new_squared_gradient_estimate])
        
        # calculate biased squared gradient
        self.new_biased_squared_gradient = ((self.squared_gradient_history[-1])/(1-((self.momentum_decay_two)**self.run_count)))
        
        self.biased_squared_gradient_history = np.append(self.biased_squared_gradient_history, [self.new_biased_squared_gradient])

    def optimize_count(self):
        derivatives = self.calc_derivatives()

        self.calc_estimated_momentum() # calc estimated biased and unbaised momentum estimates 
        self.calc_squared_grad() # calc estimated biased and unbaised squared gradient estimates 

        if np.abs(self.focus_learning_rate_history[-1] * derivatives["focus"]) > 1:
            
            self.new_focus = self.focus_history[-1] - ((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))
            
            self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
            self.new_focus = round(self.new_focus)

            self.focus_history = np.append(self.focus_history, [self.new_focus])
            mirror_values[0] = self.new_focus

        if np.abs(self.second_dispersion_learning_rate_history[-1] * derivatives["second_dispersion"]) > 1:
                                        
            self.new_second_dispersion = self.second_dispersion_history[-1] - ((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))

            self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
            self.new_second_dispersion = round(self.new_second_dispersion)

            self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])
            dispersion_values[0] = self.new_second_dispersion

    def process_images(self):
        self.run_count += 1

        self.iteration_data = np.append(self.iteration_data, [self.run_count])
        
        if self.run_count == 1 or self.run_count == 2:

            if self.run_count == 1:
                print('-------------')      

                self.focus_history = np.append(self.focus_history, [self.initial_focus])      
                self.second_dispersion_history = np.append(self.second_dispersion_history, [self.initial_second_dispersion])                   

                self.momentum_estimate_history = np.append(self.momentum_estimate_history, [self.initial_momentum_estimate])
                self.squared_gradient_history = np.append(self.squared_gradient_history, [self.initial_squared_gradient])

                self.focus_learning_rate_history = np.append(self.focus_learning_rate_history, [self.initial_focus_learning_rate])
                self.second_dispersion_learning_rate_history = np.append(self.second_dispersion_learning_rate_history, [self.initial_second_dispersion_learning_rate])
        
            if self.run_count == 2:
                self.new_focus = self.focus_history[-1] +1
                self.new_second_dispersion =  self.second_dispersion_history[-1] +1

                self.focus_history = np.append(self.focus_history, [self.new_focus])
                self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])

            self.count_function()   
            self.calc_derivatives()
            print(f"count {self.count_history[-1]}, focus = {self.focus_history[-1]}, disp2 = {self.second_dispersion_history[-1]}")

        if self.run_count > 2:
            self.count_function()   
            self.optimize_count()

            print(f"count {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}")

        self.write_values()
        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.focus_curve.setData(self.iteration_data, self.focus_der_history)
        self.second_dispersion_curve.setData(self.iteration_data, self.second_dispersion_der_history)
        self.total_gradient_curve.setData(self.iteration_data, self.total_gradient_history)

        print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])

    for _ in range(100):
        app.process_images()

    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())