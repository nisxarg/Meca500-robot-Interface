import cv2
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class CameraFeedWidget(QWidget):
    def __init__(self, cam_index, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index)
        self.label = QLabel(f"Camera {cam_index}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.label.width(), self.label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()