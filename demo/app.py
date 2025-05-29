import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import QTimer, Qt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎥 Hand Gesture Recognition - ProVIP Edition")
        self.setGeometry(100, 100, 900, 650)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QLabel#resultLabel {
                font-size: 28px;
                font-weight: bold;
                color: #00E676;
            }
            QLabel#videoLabel {
                border-radius: 15px;
                border: 3px solid #00E676;
                background-color: #222222;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 10px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: 600;
                min-width: 100px;
                transition: background-color 0.3s ease;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
            QHBoxLayout {
                spacing: 20px;
            }
        """)

        # Load model & labels
        self.model = load_model('3dcnn_lstm_m3dcnn_lstm_model_22_5odel.keras')
        self.label_map = {
            0: 'No gesture 🙅',
            1: 'Shaking Hand 🤝',
            2: 'Swiping Up 👆',
            3: 'Thumb Down 👎',
            4: 'Thumb Up 👍',
            5: 'Zooming Out With Two Fingers 🤏'
        }

        self.N_FRAMES = 37
        self.WIDTH, self.HEIGHT = 100, 100
        self.to_predict = []

        # Setup UI
        self.video_label = QLabel(self)
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(720, 540)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("📷 Camera feed will appear here")
        self.video_label.setStyleSheet("font-size: 20px; color: #777777;")

        self.result_label = QLabel("Prediction: Waiting...")
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setContentsMargins(0, 15, 0, 15)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        h_layout = QHBoxLayout()
        h_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        h_layout.addWidget(self.start_button)
        h_layout.addWidget(self.stop_button)
        h_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)
        layout.addLayout(h_layout)

        self.setLayout(layout)

        # Timer for webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.last_label = "Waiting..."
        self.last_confidence = 0.0

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # Bạn có thể đổi thành iVCam index ở đây
        if not self.cap.isOpened():
            self.result_label.setText("❌ Cannot open camera")
            return
        self.timer.start(30)  # ~30 fps
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.result_label.setText("Prediction: Waiting...")

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.result_label.setText("🛑 Prediction: Stopped")
        self.video_label.clear()
        self.to_predict.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.WIDTH, self.HEIGHT))
        self.to_predict.append(resized)

        if len(self.to_predict) == self.N_FRAMES:
            input_data = np.array(self.to_predict, dtype=np.float32) / 255.0
            input_data = input_data.reshape(1, self.N_FRAMES, self.HEIGHT, self.WIDTH, 1)

            preds = self.model.predict(input_data)
            class_id = np.argmax(preds)
            confidence = preds[0][class_id]

            self.last_label = self.label_map[class_id]
            self.last_confidence = confidence

            self.to_predict = []
            self.result_label.setText(f"Prediction: {self.last_label} ({self.last_confidence * 100:.1f}%)")

            '''self.result_label.setText(f"Prediction: {self.last_label}")'''

        # Convert frame to QImage for PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GestureApp()
    win.show()
    sys.exit(app.exec())
