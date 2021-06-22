#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
# Script was written by Johannes Lorper and Michael Schmidt

import sys
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.flowchart import Flowchart, Node
from DIPPID_pyqtnode import DIPPIDNode, BufferNode
import pyqtgraph.flowchart.library as fclib

def init_accel_nodes(layout, fc, dippid_node):
    # create buffer nodes
    buffer_node_x = fc.createNode("Buffer", pos=(150, 0))
    buffer_node_y = fc.createNode("Buffer", pos=(150, 100))
    buffer_node_z = fc.createNode("Buffer", pos=(150, 200))

    # connect nodes
    fc.connectTerminals(dippid_node['accelX'], buffer_node_x['dataIn'])
    fc.connectTerminals(dippid_node['accelY'], buffer_node_y['dataIn'])
    fc.connectTerminals(dippid_node['accelZ'], buffer_node_z['dataIn'])

if __name__ == "__main__":
    # TODO: register fft and svm nodes
    # fclib.registerNodeType(NormalVectorNode, [("Data",)])
    # fclib.registerNodeType(LogNode, [("Data",)])
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

    init_accel_nodes(layout, fc, dippid_node)

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        sys.exit(app.exec_())
