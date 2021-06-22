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

# TODO: Fix bug, frequency dict from FFT node None?
class SvmNode(Node):
    nodeName = "SvmNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "sample_input": dict(io="in"),
            "prediction": dict(io="out"),
        })

    def ctrlWidget(self):
        return self.ui

    def process(self, **kwds):
        print(kwds)
        prediction = kwds["sample_input"]
        print(prediction)


# custom text node for prediction showcasing
# TODO: node is still connect to accel_x output for testing
class DisplayTextNode(Node):
    nodeName = "DisplayTextNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "input": dict(io="in"),
            "prediction": dict(io="out"),
        })
        self.init_ui()

    def init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.text = QtGui.QLabel()
        self.text.setText("")
        self.layout.addWidget(self.text)
        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        return self.ui

    def process(self, **kwds):
        prediction = kwds["input"][0]
        self.text.setText("Prediction: " + str(prediction))


# custom FFT node for frequency spectrogram output
class FftNode(Node):
    nodeName = "FftNode"

    def __init__(self, name):
        Node.__init__(self, name, terminals={
            "accelX": dict(io="in"),
            "accelY": dict(io="in"),
            "accelZ": dict(io="in"),
            "frequency": dict(io="out"),
        })

    def calculate_frequency(self, data):
        try:
            n = len(data)
            # fft computing and normalization and
            # use only first half as the function is mirrored
            frequenzy = np.abs(np.fft.fft(data) / n)[0:int(n / 2)]
            # tolist() to convert from np.ndarray
            return frequenzy.tolist()
        except Exception as e:
            print(e)

    def process(self, **kwds):
        x_frequency = self.calculate_frequency(kwds["accelX"])
        y_frequency = self.calculate_frequency(kwds["accelY"])
        z_frequency = self.calculate_frequency(kwds["accelZ"])

        frequencies = {
            "x": x_frequency,
            "y": y_frequency,
            "z": z_frequency,
        }

        return frequencies


def init_nodes(fc, dippid_node, fft_node, svm_node, display_text_node):
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
    fc.connectTerminals(fft_node["frequency"], svm_node["sample_input"])

    # connect display text node
    fc.connectTerminals(buffer_node_x["dataOut"], display_text_node["input"])


if __name__ == "__main__":
    fclib.registerNodeType(FftNode, [("FftNode",)])
    fclib.registerNodeType(SvmNode, [("SvmNode",)])
    fclib.registerNodeType(DisplayTextNode, [("DisplayTextNode",)])

    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle("DIPPID Activity Recognizer")

    # Define a top-level widget to hold everything
    central_widget = QtGui.QWidget()
    win.setCentralWidget(central_widget)

    # Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    central_widget.setLayout(layout)

    # Creating flowchart
    fc = Flowchart(terminals={})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    # create DIPPID node
    dippid_node = fc.createNode("DIPPID", pos=(0, 0))

    # create FFT node
    fft_node = fc.createNode("FftNode", pos=(300, 100))

    # create FFT node
    svm_node = fc.createNode("SvmNode", pos=(450, 100))

    # create display text node
    display_text_node = fc.createNode("DisplayTextNode", pos=(600, 100))

    init_nodes(fc, dippid_node, fft_node, svm_node, display_text_node)

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        sys.exit(app.exec_())
