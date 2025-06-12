from PyQt6.QtWidgets import QWidget, QGridLayout
from camera_feed_widget import CameraFeedWidget

class CameraGridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout()
        self.cameras = [CameraFeedWidget(i) for i in range(4)]
        positions = [(0,0),(0,1),(1,0),(1,1)]
        for cam, pos in zip(self.cameras, positions):
            self.grid.addWidget(cam, *pos)
            cam.mouseDoubleClickEvent = self.make_maximize_handler(cam)
        self.setLayout(self.grid)
        self.maximized = False
        self.maximized_widget = None

    def make_maximize_handler(self, cam_widget):
        def handler(event):
            if not self.maximized:
                self.maximize_camera(cam_widget)
            else:
                self.restore_grid()
        return handler

    def maximize_camera(self, cam_widget):
        for cam in self.cameras:
            cam.setVisible(False)
        cam_widget.setVisible(True)
        self.layout().addWidget(cam_widget, 0, 0, 2, 2)
        self.maximized = True
        self.maximized_widget = cam_widget

    def restore_grid(self):
        for idx, cam in enumerate(self.cameras):
            cam.setVisible(True)
            row, col = divmod(idx, 2)
            self.layout().addWidget(cam, row, col)
        self.maximized = False
        self.maximized_widget = None