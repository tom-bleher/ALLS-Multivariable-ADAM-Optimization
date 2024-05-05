import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets
import sys 
import pyqtgraph as pg

# the txt files the code adjusts and uploads 
MIRROR_FILE_PATH = r'dm_parameters.txt'
DISPERSION_FILE_PATH = r'dazzler_parameters.txt'

# open and read the txt files and read the initial values
with open(MIRROR_FILE_PATH, 'r') as file:
    content = file.read()
mirror_values = list(map(int, content.split()))

with open(DISPERSION_FILE_PATH, 'r') as file:
    content = file.readlines()

dispersion_values = {
    0: int(content[0].split('=')[1].strip()),  # 0 is the key for 'order2'
    1: int(content[1].split('=')[1].strip())   # 1 is the key for 'order3'
}

class BetatronApplication(QtWidgets.QApplication):
    def __init__(self, *args, **kwargs):
        super(BetatronApplication, self).__init__(*args, **kwargs)

        # for how many images should the mean be taken for
        self.images_processed = 0
        self.count_history = np.array([])

        # set rates for optimization parameters
        self.epsilon = 1e-8
        self.momentum_decay_one = 0.9
        self.momentum_decay_two = 0.999
        self.initial_focus_learning_rate = 320
        self.initial_second_dispersion_learning_rate = 320

        self.initial_momentum_estimate = 0
        self.initial_squared_gradient = 0 

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
        self.count_plot_widget.setWindowTitle('Count optimization')
        self.count_plot_widget.setLabel('left', 'Count')
        self.count_plot_widget.setLabel('bottom', 'Image group iteration')
        self.count_plot_widget.show()

        self.main_plot_window = pg.GraphicsLayoutWidget()
        self.main_plot_window.show()

        layout = self.main_plot_window.addLayout(row=0, col=0)

        self.count_plot_widget = layout.addPlot(title='Count vs image group iteration')
        self.total_gradient_plot = layout.addPlot(title='Total gradient vs image group iteration')

        self.plot_curve = self.count_plot_widget.plot(pen='r')
        self.total_gradient_curve = self.total_gradient_plot.plot(pen='y', name='total gradient')
        
        # y labels of plots
        self.total_gradient_plot.setLabel('left', 'Total Gradient')
        self.count_plot_widget.setLabel('left', 'Image Group Iteration')

        # x label of both plots
        self.count_plot_widget.setLabel('bottom', 'Image Group Iteration')
        self.total_gradient_plot.setLabel('bottom', 'Image Group Iteration')

        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.total_gradient_curve.setData(self.der_iteration_data, self.total_gradient_history)

    # ------------ Deformable mirror ------------ #

        self.initial_focus = -240
        self.focus_history = []    
        # self.FOCUS_LOWER_BOUND = max(self.initial_focus - 20, -200)
        # self.FOCUS_UPPER_BOUND = min(self.initial_focus + 20, 200)

        self.FOCUS_LOWER_BOUND = -99999
        self.FOCUS_UPPER_BOUND = +99999

        self.tolerance = 1
        
    # ------------ Dazzler ------------ #

        self.initial_second_dispersion = -240
        self.second_dispersion_history = []
        # self.SECOND_DISPERSION_LOWER_BOUND = max(self.initial_second_dispersion - 500, 30000)
        # self.SECOND_DISPERSION_UPPER_BOUND = min(self.initial_second_dispersion + 500, 40000)

        self.SECOND_DISPERSION_LOWER_BOUND = -99999
        self.SECOND_DISPERSION_UPPER_BOUND = +99999

    def write_values(self):

        self.new_focus = round(np.clip(self.focus_history[-1], self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND))
        self.new_second_dispersion = round(np.clip(self.second_dispersion_history[-1], self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND))

        mirror_values[0] = self.new_focus
        dispersion_values[0] = self.new_second_dispersion

        with open(MIRROR_FILE_PATH, 'w') as file:
            file.write(' '.join(map(str, mirror_values)))

        with open(DISPERSION_FILE_PATH, 'w') as file:
            file.write(f'order2 = {dispersion_values[0]}\n')
            file.write(f'order3 = {dispersion_values[1]}\n')

        QtCore.QCoreApplication.processEvents()

    def count_function(self, focus_history, new_second_dispersion):
        x = focus_history
        y = new_second_dispersion

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

        return {
            "focus": self.count_focus_der,
            "second_dispersion": self.count_second_dispersion_der,
            }
    
    def calc_estimated_momentum(self):
        
        # calculate unbiased momentum estimate
        self.new_momentum_estimate = (self.momentum_decay_one*self.momentum_estimate_history[-1]) + ((1-self.momentum_decay_one)*(self.focus_der_history[-1]))
        
        self.momentum_estimate_history = np.append(self.momentum_estimate_history, [self.new_momentum_estimate])
        
        # calculate biased momentum estimate
        self.new_biased_momentum = ((self.momentum_estimate_history[-1])/(1-((self.momentum_decay_one)**self.images_processed)))
        
        self.biased_momentum_estimate_history = np.append(self.biased_momentum_estimate_history, [self.new_biased_momentum])
        

    def calc_squared_grad(self):
        
        # calculate unbiased squared gradient
        self.new_squared_gradient_estimate = (self.momentum_decay_two*self.squared_gradient_history[-1]) + ((1-self.momentum_decay_two)*((self.focus_der_history[-1])**2))
        
        self.squared_gradient_history = np.append(self.squared_gradient_history, [self.new_squared_gradient_estimate])
        
        # calculate biased squared gradient
        self.new_biased_squared_gradient = ((self.squared_gradient_history[-1])/(1-((self.momentum_decay_two)**self.images_processed)))
        
        self.biased_squared_gradient_history = np.append(self.biased_squared_gradient_history, [self.new_biased_squared_gradient])

    def optimize_count(self):
        # take the derivatives
        self.calc_derivatives()

        self.calc_estimated_momentum() # calc estimated biased and unbaised momentum estimates 
        self.calc_squared_grad() # calc estimated biased and unbaised squared gradient estimates 

        if np.abs(((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) > 1:
            
            self.new_focus = self.focus_history[-1] - ((self.focus_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))
            
            self.new_focus = np.clip(self.new_focus, self.FOCUS_LOWER_BOUND, self.FOCUS_UPPER_BOUND)
            self.new_focus = round(self.new_focus)

            self.focus_history = np.append(self.focus_history, [self.new_focus])
            mirror_values[0] = self.new_focus

        if np.abs(((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))) > 1:
                                        
            self.new_second_dispersion = self.second_dispersion_history[-1] - ((self.second_dispersion_learning_rate_history[-1]*self.biased_momentum_estimate_history[-1])/(np.sqrt(self.biased_squared_gradient_history[-1])+self.epsilon))

            self.new_second_dispersion = np.clip(self.new_second_dispersion, self.SECOND_DISPERSION_LOWER_BOUND, self.SECOND_DISPERSION_UPPER_BOUND)
            self.new_second_dispersion = round(self.new_second_dispersion)

            self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])
            dispersion_values[0] = self.new_second_dispersion

    def process_images(self):
        self.images_processed += 1
        self.iteration_data = np.append(self.iteration_data, [self.images_processed])

        if self.images_processed == 1:
                   
            self.focus_history = np.append(self.focus_history, [self.initial_focus])      
            self.second_dispersion_history = np.append(self.second_dispersion_history, [self.initial_second_dispersion])                   

            self.momentum_estimate_history = np.append(self.momentum_estimate_history, [self.initial_momentum_estimate])
            self.squared_gradient_history = np.append(self.squared_gradient_history, [self.initial_squared_gradient])

            self.focus_learning_rate_history = np.append(self.focus_learning_rate_history, [self.initial_focus_learning_rate])
            self.second_dispersion_learning_rate_history = np.append(self.second_dispersion_learning_rate_history, [self.initial_second_dispersion_learning_rate])
    
            self.count_function(self.focus_history[-1], self.second_dispersion_history[-1])   
            self.calc_derivatives()
            print(f"count {self.count_history[-1]}, focus = {self.focus_history[-1]}, disp2 = {self.second_dispersion_history[-1]}")

        elif self.images_processed == 2:
            self.new_focus = self.focus_history[-1] +1
            self.new_second_dispersion =  self.second_dispersion_history[-1] +1

            self.focus_history = np.append(self.focus_history, [self.new_focus])
            self.second_dispersion_history = np.append(self.second_dispersion_history, [self.new_second_dispersion])

            self.count_function(self.focus_history[-1], self.second_dispersion_history[-1])   
            self.calc_derivatives()
            print(f"count {self.count_history[-1]}, focus = {self.focus_history[-1]}, disp2 = {self.second_dispersion_history[-1]}")

        else:
            self.count_function(self.focus_history[-1], self.second_dispersion_history[-1])   
            self.optimize_count()

            print(f"count {self.count_history[-1]}, current values are: focus {self.focus_history[-1]}, second_dispersion {self.second_dispersion_history[-1]}")

        self.write_values()
        self.plot_curve.setData(self.iteration_data, self.count_history)
        self.total_gradient_curve.setData(self.iteration_data, self.total_gradient_history)

        print('-------------')

if __name__ == "__main__":
    app = BetatronApplication([])

    for _ in range(50):
        app.process_images()

    win = QtWidgets.QMainWindow()
    sys.exit(app.exec_())