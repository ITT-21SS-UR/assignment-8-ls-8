#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
# Script was written by Johannes Lorper and Michael Schmidt

import sys
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.flowchart import Flowchart, Node
import numpy as np
from DIPPID_pyqtnode import DIPPIDNode, BufferNode
import pyqtgraph.flowchart.library as fclib
import pandas as pd
import os
from sklearn import svm
from numpy import fromstring


'''
Custom SVM node which can be switched between
inactive, traning and prediction.

Inactive: Do nothing

Traning mode: Continually read in sample data (in our case
a list of frequency components) and trains a SVM classifier
with the data (and previous data) (Note, he category for this sample 
can be defined by a text field in the control pane)

Prediction: SVM node reads sample in and outputs the predicted category
as string. 
'''
# Filename where our gestures are saved
TRAINING_DATA_FILE = "training_data.csv"

# The amount of transformed signals from the dippid we use for 1 gesture; the transfomation cuts the dippid singal amount in half
# to calculate to time per gesture do:
# DATA_LENGTH / DippidFrequency/ * 2
DATA_LENGTH = 120


class SvmNode(Node):
    nodeName = "SvmNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "sample_input": dict(io="in"),
            "prediction": dict(io="out"),
        })


    def process(self, **kwds):
        prediction = kwds["sample_input"]
        return {'prediction': np.array(prediction)}

class PredictionNode(Node):
    nodeName = "PredictionNode"
    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "accelerator_x": dict(io="in"),
            "accelerator_y": dict(io="in"),
            "accelerator_z": dict(io="in")
        })
        self.training_data_dict = {}
        self.init_ui()
        self.init_svm()
        self.current_gesture_x_frequencies = []
        self.current_gesture_y_frequencies = []
        self.current_gesture_z_frequencies = []


    def init_svm(self):
        self.classifier = svm.SVC()
        self.update_saved_training_data()
        categories = []
        training_data = []
        if len(self.training_data_dict) > 1:
            current_index = 0
            for key, value in self.training_data_dict.items():
                categories += [current_index]
                current_values_array = []
                current_values_array.append(value.get("x"))
                current_values_array.append(value.get("y"))
                current_values_array.append(value.get("z"))

                training_data += self.get_svm_data_array(current_values_array)

                current_index += 1
            self.classifier.fit(training_data, categories)

    def get_svm_data_array(self, x_y_z_array):
        svm_data_array = []
        for value in x_y_z_array:
                x_cut = []
                for index, x_value in enumerate(x_y_z_array[0]):
                    if index < DATA_LENGTH:
                        x_cut.append(x_value)
                y_cut = []
                for index, y_value in enumerate(x_y_z_array[1]):
                    if index < DATA_LENGTH:
                        y_cut.append(y_value)
                z_cut = []
                for index, z_value in enumerate(x_y_z_array[2]):
                    if index < DATA_LENGTH:
                        z_cut.append(z_value)

                svm_data_array += x_cut +y_cut + z_cut
        return [svm_data_array]



    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.predict_button = QtGui.QPushButton()
        self.predict_button.clicked.connect(self.predict)
        self.prediction_output = QtGui.QLabel()
        self.prediction_output.setText("Predicted Gesture: None")
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.prediction_output)

        self.predict_button.setText("Start Prediction")
        self.ui.setLayout(self.layout)

    def update_saved_training_data(self):
        if os.path.isfile(TRAINING_DATA_FILE):
            saved_training_data = pd.read_csv(TRAINING_DATA_FILE)
            for index, row in saved_training_data.iterrows():
                gesture_name = row[0]
                gesture_x_frequencies = fromstring(row[1], sep="|")
                gesture_y_frequencies = fromstring(row[2], sep="|")
                gesture_z_frequencies = fromstring(row[3], sep="|")
                self.training_data_dict[gesture_name] = {"x": gesture_x_frequencies, "y": gesture_y_frequencies, "z": gesture_z_frequencies}

    def predict(self):
        input_data = []
        input_data.append(self.current_gesture_x_frequencies)
        input_data.append(self.current_gesture_y_frequencies)
        input_data.append(self.current_gesture_z_frequencies)
        predicition_data = self.get_svm_data_array(input_data)

        print(list(self.training_data_dict.keys())[self.classifier.predict(predicition_data)[0]])
        self.prediction_output.setText(list(self.training_data_dict.keys())[self.classifier.predict(predicition_data)[0]])

    def ctrlWidget(self):
        return self.ui

    def process(self, **kwds):
        # Get the last values from our accelerator data
        self.current_gesture_x_frequencies = kwds["accelerator_x"]
        self.current_gesture_y_frequencies = kwds["accelerator_y"]
        self.current_gesture_z_frequencies = kwds["accelerator_z"]



class TrainNode(Node):
    nodeName = "TrainNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "accelerator_x": dict(io="in"),
            "accelerator_y": dict(io="in"),
            "accelerator_z": dict(io="in")
        })
        self.init_ui()
        self.isRecording = False
        self.current_gesture_x_frequencies = []
        self.current_gesture_y_frequencies = []
        self.current_gesture_z_frequencies = []


    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.start_button = QtGui.QPushButton()
        self.start_button.clicked.connect(self.start_or_stop_recording)
        self.name_input = QtGui.QLineEdit()
        self.name_input.setText("Set your activity name here")
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.start_button)
        self.start_button.setText("Start Training")
        self.ui.setLayout(self.layout)

    def init_logger(self, filename):
        self.current_filename = filename
        if os.path.isfile(filename):
            self.gesture_data = pd.read_csv(filename)
        else:
            self.gesture_data = pd.DataFrame(
                columns=['gestureName', 'frequenciesX', 'frequenciesY', 'frequenciesZ'])

    def start_or_stop_recording(self):
        if self.isRecording:
            # If we were recording, we now want to stop the recording and write our pd dataframe to a csv file
            self.gesture_data = self.gesture_data.append({'gestureName':self.name_input.text(),
                                                                        'frequenciesX': "|".join(map(str,self.current_gesture_x_frequencies)),
                                                                        'frequenciesY': "|".join(map(str,self.current_gesture_y_frequencies)),
                                                                        'frequenciesZ': "|".join(map(str,self.current_gesture_z_frequencies))}
                                             , ignore_index=True)
            self.gesture_data.to_csv(self.current_filename, index=False)
            self.isRecording = False
            self.start_button.setText("Start Training")
        else:
            # If we weren't recording before we want start a pd dataframe
            self.init_logger(TRAINING_DATA_FILE)
            self.isRecording = True
            self.start_button.setText("Stop Training")

    def ctrlWidget(self):
        return self.ui

    def process(self, **kwds):
        # Get the last values from our accelerator data
        self.current_gesture_x_frequencies = kwds["accelerator_x"]
        self.current_gesture_y_frequencies = kwds["accelerator_y"]
        self.current_gesture_z_frequencies = kwds["accelerator_z"]

        # Write them to our pd dataframe if we are currently recording




# custom FFT node for frequency spectrogram output
class FftNode(Node):
    nodeName = "FftNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "accelX": dict(io="in"),
            "accelY": dict(io="in"),
            "accelZ": dict(io="in"),
            "setActive": dict(io="in"),
            "frequencyX": dict(io="out"),
            "frequencyY": dict(io="out"),
            "frequencyZ": dict(io="out"),
        })
        self.current_data_x = []
        self.current_data_y = []
        self.current_data_z = []

    def calculate_frequency(self, data):
        try:
            #  we only want to get [DATA_LENGTH] frequencies from the last signals. Since our forier
            #  transformation cuts the data amount throught 2 we use len(data)/2 for this
            while len(data)/2 > DATA_LENGTH:
                data = data[1:]
            n = len(data)
            # fft computing and normalization and
            # use only first half as the function is mirrored
            frequenzy = np.abs(np.fft.fft(data) / n)[0:int(n / 2)]

            # tolist() to convert from np.ndarray
            return frequenzy.tolist()
        except Exception as e:
            print(e)

    def process(self, **kwds):

        self.current_data_x.append(kwds["accelX"][-1])
        self.current_data_y.append(kwds["accelY"][-1])
        self.current_data_z.append(kwds["accelZ"][-1])
        x_frequency = self.calculate_frequency(self.current_data_x)
        y_frequency = self.calculate_frequency(self.current_data_y)
        z_frequency = self.calculate_frequency(self.current_data_z)

        return {'frequencyX': np.array(x_frequency), 'frequencyY': np.array(y_frequency), 'frequencyZ': np.array(z_frequency)}


def init_nodes():
    # create buffer nodes
    buffer_node_x = fc.createNode("Buffer", pos=(150, 0))
    buffer_node_y = fc.createNode("Buffer", pos=(150, 100))
    buffer_node_z = fc.createNode("Buffer", pos=(150, 200))

    # connect buffer nodes
    fc.connectTerminals(dippid_node["accelX"], buffer_node_x["dataIn"])
    fc.connectTerminals(dippid_node["accelY"], buffer_node_y["dataIn"])
    fc.connectTerminals(dippid_node["accelZ"], buffer_node_z["dataIn"])

    # connect FFT node
    fc.connectTerminals(buffer_node_x["dataOut"], fft_node["accelX"])
    fc.connectTerminals(buffer_node_y["dataOut"], fft_node["accelY"])
    fc.connectTerminals(buffer_node_z["dataOut"], fft_node["accelZ"])

    # connect svm node
    fc.connectTerminals(fft_node["frequencyY"], svm_node["sample_input"])


    # connect train node to accelerator values
    fc.connectTerminals(train_node["accelerator_x"], fft_node["frequencyX"])
    fc.connectTerminals(train_node["accelerator_y"], fft_node["frequencyY"])
    fc.connectTerminals(train_node["accelerator_z"], fft_node["frequencyZ"])
    fc.connectTerminals(prediction_node["accelerator_x"], fft_node["frequencyX"])
    fc.connectTerminals(prediction_node["accelerator_y"], fft_node["frequencyY"])
    fc.connectTerminals(prediction_node["accelerator_z"], fft_node["frequencyZ"])


if __name__ == "__main__":
    fclib.registerNodeType(FftNode, [("FftNode",)])
    fclib.registerNodeType(SvmNode, [("SvmNode",)])
    fclib.registerNodeType(TrainNode, [("TrainNode",) ])
    fclib.registerNodeType(PredictionNode, [("PredictionNode",)])

    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle("DIPPID Activity Recognizer")

    # Define a top-level widget to hold everything
    central_widget = QtGui.QWidget()
    central_widget.setFixedWidth(500)
    win.setCentralWidget(central_widget)


    # Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    ''' button_layout = QtGui.QHBoxLayout(win)
    train_button = QtGui.QPushButton("Train Gesture", win)
    recognize_button = QtGui.QPushButton("Recognize Gesture", win)
    button_layout.addWidget(train_button)
    button_layout.addWidget(recognize_button)'''

    central_widget.setLayout(layout)
    # Creating flowchart
    fc = Flowchart(terminals={})
    layout.addWidget(fc.widget(), 0, 0, 2, 3)


    # create DIPPID node
    dippid_node = fc.createNode("DIPPID", pos=(0, 0))

    # create Train node

    train_node = fc.createNode("TrainNode", pos=(450, 150))

    # create Prediction node

    prediction_node = fc.createNode("PredictionNode", pos=(450, 150))

    # create FFT node
    fft_node = fc.createNode("FftNode", pos=(300, 100))

    # create SVM node
    svm_node = fc.createNode("SvmNode", pos=(450, 100))


    init_nodes()


    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        sys.exit(app.exec_())
