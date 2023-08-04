from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import cv2
from thread import Thread

class Video(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initTabInfor()
        self.createToolBarV()

    def initUI(self):
        self.setWindowTitle("Video")
        self.setGeometry(100, 100, 400, 300)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView()
        self.view.setScene(self.scene)
        self.transform = self.view.transform()
        self.layout = QHBoxLayout()
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.view)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.splitter, stretch = 5)
        self.editToolBarV = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea,self.editToolBarV)
        self.tabs = QTabWidget()
        self.tabInformation = QWidget()
        self.tabs.addTab(self.tabInformation, 'Information')
        self.layout.addWidget(self.tabs, stretch = 1)
        self.centralWidget.setLayout(self.layout)
        self.showMaximized()

    def initTabInfor(self):
        self.information = QVBoxLayout()
        self.nameCar, widgetName = self._tool('Name: ','')
        self.lengthCar, widgetLength = self._tool('Length: ', 'cm')
        self.widthCar, widgetWidth = self._tool('Width: ', 'cm')
        self.heightCar, widgetHeight = self._tool('Height: ', 'cm')
        self.information.addWidget(widgetName)
        self.information.addWidget(widgetLength)
        self.information.addWidget(widgetWidth)
        self.information.addWidget(widgetHeight)
        self.tabInformation.setLayout(self.information)

    def _tool(self, name, cm):
        widgetT = QWidget()
        hbox = QHBoxLayout()
        widgetT.setFixedHeight(50)
        labelT = QLabel(name)
        labelDram = QLabel(cm)
        space = QLabel()
        text = QLineEdit()
        text.setAlignment(Qt.AlignCenter)
        text.setReadOnly(True)
        hbox.addWidget(labelT)
        hbox.addWidget(text)
        hbox.addWidget(labelDram)
        hbox.addWidget(space)
        widgetT.setLayout(hbox)
        return text, widgetT
    
    def createToolBarV(self):
        self.threadVideo = Thread(self)
        self.threadCamera = Thread(self)
        self.openVideo = self._createToolBar('assets/video-icon.png', self.Video, "Ctrl+O")
        # self.openCamera = self._createToolBar('assets/512px-Video_camera_icon.svg.png', self.Camera, "Ctrl+S")      
        # self.listTool = [self.openVideo, self.openCamera]
        self.listTool = [self.openVideo]
        for bt in self.listTool:
            bt.clicked.connect(self.button_clicked)
    
    def _createToolBar(self, name, log, shortCut):
        window = QWidget()
        button = QVBoxLayout()
        toolButton = QToolButton(self)
        toolButton.setAutoRaise(True)
        toolButton.clicked.connect(log)       
        toolButton.setShortcut(QKeySequence(shortCut))
        toolButton.setIcon(QIcon(name))
        toolButton.setIconSize(QSize(25, 25))
        toolButton.setCheckable(True)
        button.addWidget(toolButton)
        spacer = QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.addItem(spacer)
        window.setLayout(button)
        self.editToolBarV.addWidget(window)
        return toolButton
    
    def button_clicked(self):        
        sender = self.sender()
        sender.setChecked(True)
        for button in self.listTool:
            if button != sender:
                button.setChecked(False)

    def Video(self):       
        # self.scene.clear()
        file_name, _ = QFileDialog.getOpenFileName(
        self, "Open file", ".", "Image Files (*.mp4 *.avi)"
        )
        if not file_name:
            return
        self.threadVideo.cap = cv2.VideoCapture(file_name)
        self.threadVideo.changePixmap.connect(self.setImage)
        self.threadVideo.start()
        self.threadCamera.stop()

    def Camera(self):
        # self.scene.clear()
        try:
            self.threadCamera.cap = cv2.VideoCapture(0)
            self.threadCamera.changePixmap.connect(self.setImage)
            self.threadCamera.start()
            self.threadVideo.stop()
        except:
            return 

    def setImage(self, image, name, height, width, length):
        self.scene.clear()
        self.scene.setSceneRect(0, 0, image.width(), image.height())
        self.scene.addPixmap(QPixmap.fromImage(image))
        self.nameCar.setText(name)
        self.heightCar.setText(str(height))
        self.widthCar.setText(str(width))
        self.lengthCar.setText(str(length))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Video()
    win.show()
    sys.exit(app.exec_())