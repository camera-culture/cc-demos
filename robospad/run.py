import sys
import subprocess
import signal
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread

class SSHThread(QThread):
    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        # Start the subprocess and keep a reference to it
        self.process = subprocess.Popen(self.command, shell=True, preexec_fn=os.setsid)
        self.process.wait()

    def stop(self):
        if self.process and self.process.poll() is None:
            # Send SIGINT directly to the SSH process to simulate Ctrl+C
            self.process.send_signal(signal.SIGINT)
            # Wait for the process to terminate
            self.process.wait()
            self.process = None

class ControlPanel(QWidget):
    def __init__(self, ip_address):
        super().__init__()
        self.ip_address = ip_address
        self.ssh_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Control Panel')
        self.showFullScreen()  # Make the window full-screen

        # Define button styles
        button_font = QFont('Helvetica', 40, QFont.Bold)

        # Create the red button (Left Side)
        btn_red = QPushButton('LOS Only')
        btn_red.setFont(button_font)
        btn_red.setStyleSheet("background-color: #ff6961; color: white;")
        btn_red.clicked.connect(self.run_command)
        btn_red.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create the blue button (Right Side)
        btn_blue = QPushButton('NLOS Enabled')
        btn_blue.setFont(button_font)
        btn_blue.setStyleSheet("background-color: #72a4d4; color: white;")
        btn_blue.clicked.connect(self.run_command)
        btn_blue.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create the Stop button (Bottom)
        btn_stop = QPushButton('Stop')
        btn_stop.setFont(button_font)
        btn_stop.setStyleSheet("background-color: black; color: white;")
        btn_stop.clicked.connect(self.stop_command)
        btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Arrange buttons in layouts
        h_layout = QHBoxLayout()
        h_layout.addWidget(btn_red)
        h_layout.addWidget(btn_blue)
        h_layout.setSpacing(0)
        h_layout.setContentsMargins(0, 0, 0, 0)

        # Place the horizontal layout into a vertical layout
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(btn_stop)
        v_layout.setSpacing(0)
        v_layout.setContentsMargins(0, 0, 0, 0)

        # Set stretch factors to allocate more space to the main buttons
        v_layout.setStretch(0, 1)  # The h_layout (main buttons) gets all available vertical space
        v_layout.setStretch(1, 0)  # The Stop button maintains its default height

        self.setLayout(v_layout)
        self.show()

    def run_command(self):
        if self.ssh_thread and self.ssh_thread.isRunning():
            # Command is already running
            return

        command = f'ssh -t camera@{self.ip_address} "/home/camera/projects/freenove_car/env/bin/python /home/camera/projects/freenove_car/Code/Server-pi5/tracker.py"'
        self.ssh_thread = SSHThread(command)
        self.ssh_thread.start()

    def stop_command(self):
        if self.ssh_thread:
            self.ssh_thread.stop()
            self.ssh_thread.wait()
            self.ssh_thread = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Ensure the SSH command is stopped when the application exits
        self.stop_command()
        event.accept()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py IP_ADDRESS")
        sys.exit(1)

    ip_address = sys.argv[1]

    app = QApplication(sys.argv)
    window = ControlPanel(ip_address)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

