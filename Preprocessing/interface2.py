import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
from PIL import Image
import numpy as np
from skimage import transform
import keras
from keras import Model
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf


IMG_SIZE = (299,299)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(617, 457)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.SelectBtn = QtWidgets.QPushButton(self.centralwidget)
        self.SelectBtn.setGeometry(QtCore.QRect(50, 290, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.SelectBtn.setFont(font)
        self.SelectBtn.setObjectName("SelectBtn")
        self.OrigImg = QtWidgets.QLabel(self.centralwidget)
        self.OrigImg.setGeometry(QtCore.QRect(10, 30, 251, 251))
        self.OrigImg.setFrameShape(QtWidgets.QFrame.Box)
        self.OrigImg.setText("")
        self.OrigImg.setObjectName("OrigImg")
        self.HasilImg = QtWidgets.QLabel(self.centralwidget)
        self.HasilImg.setGeometry(QtCore.QRect(320, 30, 251, 251))
        self.HasilImg.setFrameShape(QtWidgets.QFrame.Box)
        self.HasilImg.setText("")
        self.HasilImg.setObjectName("HasilImg")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(380, 290, 160, 21))
        self.label.setFrameShape(QtWidgets.QFrame.Panel)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setText("")
        self.label.setObjectName("label")
        self.SelectBtn_2 = QtWidgets.QPushButton(self.centralwidget)
        self.SelectBtn_2.setGeometry(QtCore.QRect(220, 340, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.SelectBtn_2.setFont(font)
        self.SelectBtn_2.setObjectName("SelectBtn_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 0, 241, 20))
        self.label_2.setLineWidth(5)
        self.label_2.setObjectName("label_2")
        """
        self.conglbl = QtWidgets.QLabel(self.centralwidget)
        self.conglbl.setGeometry(QtCore.QRect(380, 320, 150, 21))
        self.conglbl.setFrameShape(QtWidgets.QFrame.Panel)
        self.conglbl.setFrameShadow(QtWidgets.QFrame.Plain)
        self.conglbl.setText("")
        self.conglbl.setObjectName("conglbl")
        """
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 617, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.SelectBtn.clicked.connect(self.setImage)
        self.SelectBtn_2.clicked.connect(self.Deteksi)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.SelectBtn.setText(_translate("MainWindow", "Pilih Citra X-ray Scan"))
        self.SelectBtn_2.setText(_translate("MainWindow", "DETEKSI"))
        self.label_2.setText(_translate("MainWindow", "DETEKSI COVID-19 PADA X-RAY "))
        
    def setImage(self):
        global fileName
        fileName,_=QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files(*.png *.jpg *.jpeg *.bmp *.tif)")
        if fileName:
            pixmap=QtGui.QPixmap(fileName)
            pixmap=pixmap.scaled(self.OrigImg.width(),self.OrigImg.height(),QtCore.Qt.KeepAspectRatio)
            #pixmap=pixmap.scaled(137,165,QtCore.Qt.KeepAspectRatio)
            self.OrigImg.setPixmap(pixmap)
            self.OrigImg.setAlignment(QtCore.Qt.AlignCenter)
            self.HasilImg.setPixmap(pixmap)
            self.HasilImg.setAlignment(QtCore.Qt.AlignCenter)
            print(fileName)

    def predict(model, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        return preds[0]
        
        
    def Deteksi(self):
        #img2=cv2.imread(fileName)
        #Hasil=Watershed(img2)
        #cv2.imshow("A",Hasil)
        #print(Hasil.shape)

        def predict(model, img):
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.vgg16.preprocess_input(x)
            preds = model.predict(x)
            return preds[0]
    
        img = tf.keras.preprocessing.image.load_img(fileName, target_size=IMG_SIZE)
        vgg16=tf.keras.models.load_model('MODEL_VGG16_COVID-19.h5')
        preds = predict(vgg16, img)
        result = preds[0]
        print(result)
        print(preds)
        
        if result<=0.5:
                print ("POSITIVE COVID-19")
                self.label.setText("POSITIVE COVID-19")
        elif result>0.5:
                print("NEGATIF COVID-19 (NORMAL)")
                self.label.setText("NEGATIF COVID-19 (NORMAL)")
        
        """
        #height, width, channel = img2.shape
        #img = QImage(img2,width,height, QImage.Format_RGB888)
        #pixmap = QPixmap.fromImage(img)
        pixmap = QPixmap(img2)
        self.HasilImg.setPixmap(pixmap)
        self.HasilImg.setScaledContents(True);
        """
            

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
