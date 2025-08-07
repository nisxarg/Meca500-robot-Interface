"""
Meca500 Programming Interface
----------------------------
"""
import os
import json
import time
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Union

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QLineEdit,
    QDialog, QDialogButtonBox, QFormLayout, QTabWidget,
    QDoubleSpinBox, QSpinBox, QGroupBox, QMessageBox,
    QMenu, QApplication, QFileDialog, QCheckBox, QToolTip
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QColor


# Movement type descriptions for user guidance
MOVEMENT_TYPE_DESCRIPTIONS = {
    "MoveJ": "Joint movement - Fastest motion, but path is unpredictable. Best for free movements when path doesn't matter.",
    "MoveL": "Linear movement - Robot moves in a straight line in Cartesian space. Best for precise movements around workpieces.",
    "MoveP": "Point-to-point movement - Similar to linear but with smoother acceleration. Best for continuous paths."
}


class StepDialog(QDialog):
    #Dialog for adding or editing a program step

    def __init__(self, parent=None, robot=None, step_type="move", step_data=None):
        super().__init__(parent)
        self.robot = robot
        self.step_type = step_type
        self.step_data = step_data or {}

        self.setWindowTitle(f"{'Edit' if step_data else 'Add'} Step")
        self.setMinimumWidth(400)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Step type selection
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Move to Position", "Open Gripper", "Close Gripper", "Vacuum On", "Vacuum Off","Set Gripper Position", "Wait", "Home", "Loop Start", "Loop End"])

        # Set initial selection based on step_type
        if step_type == "move":
            self.type_combo.setCurrentIndex(0)
        elif step_type == "open_gripper":
            self.type_combo.setCurrentIndex(1)
        elif step_type == "close_gripper":
            self.type_combo.setCurrentIndex(2)
        elif step_type == "set_gripper":
            self.type_combo.setCurrentIndex(3)
        elif step_type == "vacuum_on":
            self.type_combo.setCurrentIndex(4)
        elif step_type == "vacuum_off":
            self.type_combo.setCurrentIndex(5)
        elif step_type == "wait":
            self.type_combo.setCurrentIndex(6)
        elif step_type == "home":
            self.type_combo.setCurrentIndex(7)
        elif step_type == "loop_start":
            self.type_combo.setCurrentIndex(8)
        elif step_type == "loop_end":
            self.type_combo.setCurrentIndex(9)

        self.type_combo.currentIndexChanged.connect(self.update_form)
        self.main_layout.addWidget(QLabel("Step Type:"))
        self.main_layout.addWidget(self.type_combo)

        # Form container
        self.form_container = QWidget()
        self.form_layout = QVBoxLayout()
        self.form_container.setLayout(self.form_layout)
        self.main_layout.addWidget(self.form_container)

        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)

        # Initialize form based on step type
        self.update_form()

    def update_form(self):

        for i in reversed(range(self.form_layout.count())):
            widget = self.form_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Get selected type
        selected_type = self.type_combo.currentText()

        if selected_type == "Move to Position":
            self.create_move_form()
        elif selected_type == "Open Gripper":
            self.create_open_gripper_form()
        elif selected_type == "Close Gripper":
            self.create_close_gripper_form()
        elif selected_type == "Set Gripper Position":
            self.create_set_gripper_form()
        elif selected_type == "Vacuum On":
            self.create_vacuum_on_form()  # <--- NEW
        elif selected_type == "Vacuum Off":
            self.create_vacuum_off_form()
        elif selected_type == "Wait":
            self.create_wait_form()
        elif selected_type == "Home":
            self.create_home_form()
        elif selected_type == "Loop Start":
            self.create_loop_start_form()
        elif selected_type == "Loop End":
            self.create_loop_end_form()

    def create_move_form(self):

        # Movement type selection
        move_type_layout = QHBoxLayout()
        move_type_layout.addWidget(QLabel("Movement Type:"))

        self.move_type_combo = QComboBox()
        self.move_type_combo.addItems(["MoveJ", "MoveL", "MoveP"])

        # Add tooltips with descriptions
        self.move_type_combo.setToolTip(
            "<b>MoveJ</b>: Joint movement - Fastest motion, but path is unpredictable.<br>"
            "<b>MoveL</b>: Linear movement - Robot moves in a straight line.<br>" 
            "<b>MoveP</b>: Point-to-point - Similar to linear but with smoother acceleration."
        )

        # Set default or existing value
        if self.step_data and "move_type" in self.step_data:
            index = {"movej": 0, "movel": 1, "movep": 2}.get(self.step_data["move_type"].lower(), 0)
            self.move_type_combo.setCurrentIndex(index)

        move_type_layout.addWidget(self.move_type_combo)
        self.form_layout.addLayout(move_type_layout)

        # Add movement type description label
        self.move_type_description = QLabel(MOVEMENT_TYPE_DESCRIPTIONS["MoveJ"])
        self.move_type_description.setWordWrap(True)
        self.move_type_description.setStyleSheet("color: #666; font-style: italic;")
        self.form_layout.addWidget(self.move_type_description)

        # Update description when movement type changes
        self.move_type_combo.currentTextChanged.connect(self.update_move_type_description)

        # Create tabs for joint and cartesian coordinates
        self.tabs = QTabWidget()

        # Joint coordinates tab
        joint_tab = QWidget()
        joint_layout = QFormLayout()
        joint_tab.setLayout(joint_layout)

        self.joint_inputs = []
        for i in range(6):
            spin = QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)

            # Set default or existing value
            if self.step_data and "joints" in self.step_data and len(self.step_data["joints"]) > i:
                spin.setValue(self.step_data["joints"][i])

            joint_layout.addRow(f"J{i+1}:", spin)
            self.joint_inputs.append(spin)

        # Cartesian coordinates tab
        cart_tab = QWidget()
        cart_layout = QFormLayout()
        cart_tab.setLayout(cart_layout)

        self.cart_inputs = []
        cart_labels = ["X:", "Y:", "Z:", "Rx:", "Ry:", "Rz:"]
        for i, label in enumerate(cart_labels):
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)

            # Set default or existing value
            if self.step_data and "position" in self.step_data and len(self.step_data["position"]) > i:
                spin.setValue(self.step_data["position"][i])

            cart_layout.addRow(label, spin)
            self.cart_inputs.append(spin)

        self.tabs.addTab(joint_tab, "Joint Coordinates")
        self.tabs.addTab(cart_tab, "Cartesian Coordinates")

        # Set the active tab based on movement type
        self.move_type_combo.currentIndexChanged.connect(self.update_active_tab)
        self.update_active_tab(self.move_type_combo.currentIndex())

        self.form_layout.addWidget(self.tabs)

        # Record current position button
        record_btn = QPushButton("Record Current Position")
        record_btn.clicked.connect(self.record_current_position)
        self.form_layout.addWidget(record_btn)

        form_layout = QFormLayout()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 100.0)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setSingleStep(1.0)
        self.speed_spin.setSuffix(" %")
        if self.step_data and "speed" in self.step_data:
            self.speed_spin.setValue(self.step_data["speed"])
        else:
            self.speed_spin.setValue(20.0)  # Default speed

        form_layout.addRow("Speed:", self.speed_spin)
        self.form_layout.addLayout(form_layout)

    def create_vacuum_on_form(self):
        self.form_layout.addWidget(QLabel("Activate pneumatic vacuum (Vacuum ON)."))

    def create_vacuum_off_form(self):
        self.form_layout.addWidget(QLabel("Deactivate pneumatic vacuum (Vacuum OFF)."))
    def update_move_type_description(self, move_type):

        self.move_type_description.setText(MOVEMENT_TYPE_DESCRIPTIONS.get(move_type, ""))

    def update_active_tab(self, index):

        # MoveJ uses joint coordinates, MoveL and MoveP use cartesian
        if index == 0:  # MoveJ
            self.tabs.setCurrentIndex(0)  # Joint tab
        else:  # MoveL or MoveP
            self.tabs.setCurrentIndex(1)  # Cartesian tab

    def record_current_position(self):

        if not self.robot:
            return

        try:
            # Get current joint positions
            joints = self.robot.GetJoints()
            for i, val in enumerate(joints):
                if i < len(self.joint_inputs):
                    self.joint_inputs[i].setValue(val)

            # Get current cartesian position
            pose = self.robot.GetPose()
            for i, val in enumerate(pose):
                if i < len(self.cart_inputs):
                    self.cart_inputs[i].setValue(val)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to get robot position: {e}")

    def create_open_gripper_form(self):

        self.form_layout.addWidget(QLabel("Opens the gripper fully."))

    def create_close_gripper_form(self):

        self.form_layout.addWidget(QLabel("Closes the gripper fully."))

    def create_set_gripper_form(self):

        form_layout = QFormLayout()

        self.gripper_pos_spin = QSpinBox()
        self.gripper_pos_spin.setRange(0, 100)
        self.gripper_pos_spin.setSuffix("%")

        # Set default or existing value
        if self.step_data and "position" in self.step_data:
            self.gripper_pos_spin.setValue(self.step_data["position"])

        form_layout.addRow("Opening:", self.gripper_pos_spin)
        self.form_layout.addLayout(form_layout)

    def create_wait_form(self):

        form_layout = QFormLayout()

        self.wait_time_spin = QDoubleSpinBox()
        self.wait_time_spin.setRange(0.1, 60)
        self.wait_time_spin.setDecimals(1)
        self.wait_time_spin.setSingleStep(0.5)
        self.wait_time_spin.setSuffix(" seconds")

        # Set default or existing value
        if self.step_data and "time" in self.step_data:
            self.wait_time_spin.setValue(self.step_data["time"])
        else:
            self.wait_time_spin.setValue(1.0)

        form_layout.addRow("Wait Time:", self.wait_time_spin)
        self.form_layout.addLayout(form_layout)

    def create_home_form(self):

        self.form_layout.addWidget(QLabel("Moves the robot to the home position (all joints at 0)."))

    def create_loop_start_form(self):
        """Create form for Loop Start step"""
        form_layout = QFormLayout()

        self.loop_count_spin = QSpinBox()
        self.loop_count_spin.setRange(1, 1000)
        self.loop_count_spin.setSingleStep(1)
        self.loop_count_spin.setToolTip("Number of times to repeat the loop")

        # Set default or existing value
        if self.step_data and "count" in self.step_data:
            self.loop_count_spin.setValue(self.step_data["count"])
        else:
            self.loop_count_spin.setValue(2)

        form_layout.addRow("Repeat Count:", self.loop_count_spin)
        self.form_layout.addLayout(form_layout)

        # Add description
        description = QLabel("Marks the start of a loop. Must be paired with a Loop End step.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; font-style: italic;")
        self.form_layout.addWidget(description)

    def create_loop_end_form(self):
        """Create form for Loop End step"""
        description = QLabel("Marks the end of a loop. Must be paired with a Loop Start step.")
        description.setWordWrap(True)
        description.setStyleSheet("color: #666; font-style: italic;")
        self.form_layout.addWidget(description)

    def get_step_data(self):
        """Get the step data from the form"""
        selected_type = self.type_combo.currentText()

        if selected_type == "Move to Position":
            move_type = self.move_type_combo.currentText()
            joints = [spin.value() for spin in self.joint_inputs]
            position = [spin.value() for spin in self.cart_inputs]
            speed = self.speed_spin.value() if hasattr(self, "speed_spin") else 20.0
            return {
                "type": "move",
                "move_type": move_type,
                "joints": joints,
                "position": position,
                "speed": speed
            }

        elif selected_type == "Open Gripper":
            return {
                "type": "open_gripper"
            }

        elif selected_type == "Close Gripper":
            return {
                "type": "close_gripper"
            }

        elif selected_type == "Set Gripper Position":
            return {
                "type": "set_gripper",
                "position": self.gripper_pos_spin.value()
            }
        elif selected_type == "Vacuum On":
            return {"type": "vacuum_on"}  # <--- NEW
        elif selected_type == "Vacuum Off":
            return {"type": "vacuum_off"}

        elif selected_type == "Wait":
            return {
                "type": "wait",
                "time": self.wait_time_spin.value()
            }

        elif selected_type == "Home":
            return {
                "type": "home"
            }

        elif selected_type == "Loop Start":
            return {
                "type": "loop_start",
                "count": self.loop_count_spin.value(),
                "current_iteration": 0  # Will be used during execution
            }

        elif selected_type == "Loop End":
            return {
                "type": "loop_end"
            }

        return {}


class  ProgrammingInterface(QWidget):
    """Visual programming interface for the Meca500 robot"""

    # Signal to update console in main GUI
    console_update = pyqtSignal(str)

    def __init__(self, parent=None, robot=None):
        super().__init__(parent)
        self.robot = robot
        self.program_steps = []
        self.current_step_index = -1
        self.running = False
        self.program_file = None
        self.gripper_state = None
        self.loop_stack = []
        self.command_complete = True
        self.robot_state_synced = False

        self.init_ui()

        # Initialize state synchronization timer
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self.sync_robot_state)
        self.sync_timer.start(5000)  # Check every 5 seconds (reduced frequency)

        # Perform initial state synchronization
        QTimer.singleShot(500, self.sync_robot_state)

    def init_ui(self):
        """Initialize the user interface"""
        self.setMinimumHeight(300)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Title
        title_label = QLabel("Robot Programming")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # Program steps list
        self.steps_list = QListWidget()
        self.steps_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.steps_list.customContextMenuRequested.connect(self.show_context_menu)
        main_layout.addWidget(self.steps_list)

        # Buttons for program management
        buttons_layout = QHBoxLayout()

        # Left side buttons (step management)
        left_buttons = QHBoxLayout()

        self.add_step_btn = QPushButton("Add Step")
        self.add_step_btn.clicked.connect(self.add_step)
        left_buttons.addWidget(self.add_step_btn)

        self.edit_step_btn = QPushButton("Edit Step")
        self.edit_step_btn.clicked.connect(self.edit_step)
        left_buttons.addWidget(self.edit_step_btn)

        self.delete_step_btn = QPushButton("Delete Step")
        self.delete_step_btn.clicked.connect(self.delete_step)
        left_buttons.addWidget(self.delete_step_btn)

        self.run_step_btn = QPushButton("Run Step")
        self.run_step_btn.clicked.connect(self.run_selected_step)
        left_buttons.addWidget(self.run_step_btn)

        buttons_layout.addLayout(left_buttons)

        # Spacer
        buttons_layout.addStretch()

        # Right side buttons (program management)
        right_buttons = QHBoxLayout()

        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_step_up)
        right_buttons.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_step_down)
        right_buttons.addWidget(self.move_down_btn)

        self.copy_step_btn = QPushButton("Copy Step")
        self.copy_step_btn.clicked.connect(self.copy_step)
        right_buttons.addWidget(self.copy_step_btn)

        self.paste_step_btn = QPushButton("Paste Step")
        self.paste_step_btn.clicked.connect(self.paste_step)
        right_buttons.addWidget(self.paste_step_btn)

        buttons_layout.addLayout(right_buttons)

        main_layout.addLayout(buttons_layout)

        # Record position and robot control buttons
        robot_control_layout = QHBoxLayout()

        self.record_pos_btn = QPushButton("Record Position")
        self.record_pos_btn.clicked.connect(self.record_position)
        robot_control_layout.addWidget(self.record_pos_btn)

        self.true_home_btn = QPushButton("Activation & Home ")
        self.true_home_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.true_home_btn.setToolTip("Trigger the robot's true homing condition (Home command)")
        self.true_home_btn.clicked.connect(self.execute_true_home)
        robot_control_layout.addWidget(self.true_home_btn)

        main_layout.addLayout(robot_control_layout)

        # Program control buttons
        control_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Program")
        self.load_btn.clicked.connect(self.load_program)
        control_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Program")
        self.save_btn.clicked.connect(self.save_program)
        control_layout.addWidget(self.save_btn)

        self.run_btn = QPushButton("Run Program")
        self.run_btn.clicked.connect(self.run_program)
        control_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_program)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        main_layout.addLayout(control_layout)

        # Progress bar for step execution


        # Error message label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red; background-color: #ffeeee; padding: 5px;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        main_layout.addWidget(self.error_label)

        # Reset error button
        self.reset_error_btn = QPushButton("Reset Error")
        self.reset_error_btn.setStyleSheet("background-color: orange;")
        self.reset_error_btn.clicked.connect(self.reset_error)
        self.reset_error_btn.setVisible(False)
        main_layout.addWidget(self.reset_error_btn)

        self.gripper_state_label = QLabel("Gripper: Unknown")
        self.gripper_state_label.setStyleSheet("color: gray;")

        # Initialize timer for program execution
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.execute_next_step)

        # Initialize wait timer for wait steps
        self.wait_timer = QTimer()
        self.wait_timer.setSingleShot(True)
        self.wait_timer.timeout.connect(self.wait_completed)

        # Initialize command completion timer
        self.command_timer = QTimer()
        self.command_timer.setSingleShot(True)
        self.command_timer.timeout.connect(self.check_command_completion)

        # Initialize error message timer
        self.error_timer = QTimer()
        self.error_timer.setSingleShot(True)
        self.error_timer.timeout.connect(lambda: self.error_label.setVisible(False))


        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #aaa; font-style: italic;")
        main_layout.addWidget(self.status_label)
        # Update UI state
        self.update_ui_state()

    def show_context_menu(self, position):
        """Show context menu for steps list"""
        menu = QMenu()

        # Get the item at the position
        item = self.steps_list.itemAt(position)

        if item:
            # Add actions for selected item
            edit_action = QAction("Edit", self)
            edit_action.triggered.connect(self.edit_step)
            menu.addAction(edit_action)

            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(self.delete_step)
            menu.addAction(delete_action)

            run_action = QAction("Run Step", self)
            run_action.triggered.connect(self.run_selected_step)
            menu.addAction(run_action)

            menu.addSeparator()

            move_up_action = QAction("Move Up", self)
            move_up_action.triggered.connect(self.move_step_up)
            menu.addAction(move_up_action)

            move_down_action = QAction("Move Down", self)
            move_down_action.triggered.connect(self.move_step_down)
            menu.addAction(move_down_action)

            menu.addSeparator()

            copy_action = QAction("Copy", self)
            copy_action.triggered.connect(self.copy_step)
            menu.addAction(copy_action)

        # Always add paste action
        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.paste_step)
        menu.addAction(paste_action)

        # Show the menu
        menu.exec(self.steps_list.mapToGlobal(position))

    def add_step(self):
        """Add a new step to the program"""
        dialog = StepDialog(self, self.robot)
        if dialog.exec():
            step_data = dialog.get_step_data()
            self.program_steps.append(step_data)
            self.update_steps_list()
            self.steps_list.setCurrentRow(len(self.program_steps) - 1)

    def edit_step(self):
        """Edit the selected step"""
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.program_steps):
            return

        step = self.program_steps[current_row]
        dialog = StepDialog(self, self.robot, step.get("type", ""), step)

        if dialog.exec():
            self.program_steps[current_row] = dialog.get_step_data()
            self.update_steps_list()
            self.steps_list.setCurrentRow(current_row)

    def delete_step(self):
        """Delete the selected step"""
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.program_steps):
            return

        del self.program_steps[current_row]
        self.update_steps_list()

        # Select the next item or the last item if we deleted the last one
        if current_row < len(self.program_steps):
            self.steps_list.setCurrentRow(current_row)
        elif len(self.program_steps) > 0:
            self.steps_list.setCurrentRow(len(self.program_steps) - 1)

    def move_step_up(self):
        """Move the selected step up in the list"""
        current_row = self.steps_list.currentRow()
        if current_row <= 0 or current_row >= len(self.program_steps):
            return

        self.program_steps[current_row], self.program_steps[current_row - 1] = \
            self.program_steps[current_row - 1], self.program_steps[current_row]

        self.update_steps_list()
        self.steps_list.setCurrentRow(current_row - 1)

    def move_step_down(self):
        """Move the selected step down in the list"""
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.program_steps) - 1:
            return

        self.program_steps[current_row], self.program_steps[current_row + 1] = \
            self.program_steps[current_row + 1], self.program_steps[current_row]

        self.update_steps_list()
        self.steps_list.setCurrentRow(current_row + 1)

    def copy_step(self):
        """Copy the selected step to clipboard"""
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.program_steps):
            return

        step_data = self.program_steps[current_row]
        clipboard_text = f"meca500_step:{json.dumps(step_data)}"

        clipboard = QApplication.clipboard()
        clipboard.setText(clipboard_text)

        self.status_label.setText("Step copied to clipboard")

    def paste_step(self):
        """Paste a step from clipboard"""
        clipboard = QApplication.clipboard()
        text = clipboard.text()

        if not text.startswith("meca500_step:"):
            return

        try:
            step_data = json.loads(text.replace("meca500_step:", "", 1))
            self.program_steps.append(step_data)
            self.update_steps_list()
            self.steps_list.setCurrentRow(len(self.program_steps) - 1)

            self.status_label.setText("Step pasted from clipboard")
        except Exception as e:
            self.show_error(f"Error pasting step: {e}")

    def record_position(self):
        """Record the current robot position as a new step"""
        if not self.robot:
            self.show_error("Robot not connected")
            return

        try:
            # Get current joint positions
            joints = self.robot.GetJoints()
            # Get current cartesian position
            position = self.robot.GetPose()
            # Get default or last used speed
            speed = 20.0
            # Try to get from the last step or your GUI if possible
            if self.program_steps and "speed" in self.program_steps[-1]:
                speed = self.program_steps[-1]["speed"]

            # Create a new step with the current position
            step_data = {
                "type": "move",
                "move_type": "MoveL",  # Default to linear movement
                "joints": joints,
                "position": position,
                "speed": speed
            }

            self.program_steps.append(step_data)
            self.update_steps_list()
            self.steps_list.setCurrentRow(len(self.program_steps) - 1)

            self.status_label.setText("Position recorded")

        except Exception as e:
            self.show_error(f"Error recording position: {e}")
    def load_program(self):
        """Load a program from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Program", "", "Robot Program Files (*.rp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid program file format")

            self.program_steps = data
            self.program_file = file_path
            self.update_steps_list()

            # Enable Run Program button
            self.run_btn.setEnabled(len(self.program_steps) > 0)

            self.status_label.setText(f"Program loaded from {os.path.basename(file_path)}")

        except Exception as e:
            self.show_error(f"Error loading program: {e}")

    def save_program(self):
        """Save the program to a file"""
        if not self.program_steps:
            self.status_label.setText("No steps to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Program", "", "Robot Program Files (*.rp);;All Files (*)"
        )

        if not file_path:
            return

        # Add .rp extension if not present
        if not file_path.lower().endswith('.rp'):
            file_path += '.rp'

        try:
            with open(file_path, 'w') as f:
                json.dump(self.program_steps, f, indent=2)

            self.program_file = file_path
            self.status_label.setText(f"Program saved to {os.path.basename(file_path)}")

        except Exception as e:
            self.show_error(f"Error saving program: {e}")

    def run_program(self):
        """Run the entire program"""
        if not self.program_steps:
            self.status_label.setText("No steps to run")
            return

        if self.running:
            self.stop_program()  # Stop current execution before starting a new one
            # Give a small delay before restarting
            QTimer.singleShot(500, self._start_program_execution)
            return
        else:
            self._start_program_execution()

    def _start_program_execution(self):
        """Helper method to start program execution after ensuring proper state"""
        if not self.validate_loop_structure():
            return

        # Full state reset
        self.running = True
        self.current_step_index = -1
        self.loop_stack = []
        self.command_complete = True
        self.single_step_mode = False

        # Ensure all timers are properly connected
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.timer.timeout.connect(self.execute_next_step)

        try:
            self.wait_timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.wait_timer.timeout.connect(self.wait_completed)

        try:
            self.command_timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.command_timer.timeout.connect(self.check_command_completion)

        self.update_ui_state()
        self.execute_next_step()

    def validate_loop_structure(self):
        """Validate that loops are properly structured"""
        stack = []

        for i, step in enumerate(self.program_steps):
            if step.get("type") == "loop_start":
                stack.append(i)
            elif step.get("type") == "loop_end":
                if not stack:
                    self.show_error("Loop End at step {} has no matching Loop Start".format(i+1))
                    return False
                stack.pop()

        if stack:
            self.show_error("Loop Start at step {} has no matching Loop End".format(stack[0]+1))
            return False

        return True

    def run_selected_step(self):
        """Run only the selected step"""
        current_row = self.steps_list.currentRow()
        if current_row < 0 or current_row >= len(self.program_steps):
            return

        if self.running:
            self.stop_program()  # Stop current execution before starting a new one
            # Give a small delay before restarting
            QTimer.singleShot(500, lambda: self._start_selected_step(current_row))
            return
        else:
            self._start_selected_step(current_row)

    def _start_selected_step(self, current_row):
        """Helper method to start execution of a selected step"""
        # Reset state
        self.running = True
        self.loop_stack = []
        self.command_complete = True
        self.single_step_mode = True
        self.current_step_index = current_row - 1  # Will increment before execution

        # Ensure all timers are properly connected
        try:
            self.timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.timer.timeout.connect(self.execute_next_step)

        try:
            self.wait_timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.wait_timer.timeout.connect(self.wait_completed)

        try:
            self.command_timer.timeout.disconnect()
        except TypeError:
            pass  # Already disconnected
        self.command_timer.timeout.connect(self.check_command_completion)

        self.update_ui_state()
        self.execute_next_step(run_single=True)

    def execute_next_step(self, run_single=False):
        """Execute the next step in the program"""
        if not self.running or not self.program_steps:
            return
        if self.current_step_index >= len(self.program_steps):
            self.program_completed()
            return

        # Check if we need to process a loop
        if self.loop_stack and self.current_step_index == self.loop_stack[-1]["end_index"]:
            # We've reached the end of a loop
            loop_data = self.loop_stack[-1]
            loop_step = self.program_steps[loop_data["start_index"]]

            # Increment the iteration counter
            loop_data["current_iteration"] += 1

            # Check if we need to repeat
            if loop_data["current_iteration"] < loop_step.get("count", 1):
                # Jump back to the instruction after the loop start
                self.current_step_index = loop_data["start_index"]
                self.status_label.setText(f"Loop iteration {loop_data['current_iteration']+1}/{loop_step.get('count', 1)}")
            else:
                # Loop complete, remove from stack and continue
                self.loop_stack.pop()

        self.current_step_index += 1

        # Check if we've reached the end of the program
        if self.current_step_index >= len(self.program_steps):
            self.program_completed()
            return

        # Highlight the current step
        self.steps_list.setCurrentRow(self.current_step_index)

        # Get the current step
        step = self.program_steps[self.current_step_index]
        step_type = step.get("type", "")

        # Update status
        self.status_label.setText(f"Executing step {self.current_step_index + 1}: {self.get_step_description(step)}")

        try:
            # Handle loop start
            if step_type == "loop_start":
                # Find matching loop end
                end_index = self.find_matching_loop_end(self.current_step_index)
                if end_index == -1:
                    raise Exception("No matching Loop End found")

                # Add to loop stack
                self.loop_stack.append({
                    "start_index": self.current_step_index,
                    "end_index": end_index,
                    "current_iteration": 0
                })

                # Display status
                count = step.get("count", 1)
                self.status_label.setText(f"Starting loop (repeat {count} times)")

                # Continue to next step
                QTimer.singleShot(100, lambda: self.execute_next_step(run_single))
                return

            # Handle loop end
            elif step_type == "loop_end":
                # This is handled at the beginning of this method
                # Just continue to the next step or back to the loop start
                QTimer.singleShot(100, lambda: self.execute_next_step(run_single))
                return

            # Execute the step based on its type
            if step_type == "move":
                self.execute_move_step(step)
            elif step_type == "open_gripper":
                self.execute_open_gripper_step()
            elif step_type == "close_gripper":
                self.execute_close_gripper_step()
            elif step_type == "set_gripper":
                self.execute_set_gripper_step(step)
            elif step_type == "vacuum_on":
                self.execute_vacuum_on_step()  # <--- NEW
            elif step_type == "vacuum_off":
                self.execute_vacuum_off_step()
            elif step_type == "wait":
                self.execute_wait_step(step)
            elif step_type == "home":
                self.execute_home_step()
            else:
                self.show_error(f"Unknown step type: {step_type}")
                self.stop_program()
                return

            # Start the command completion timer
            self.command_complete = False


            # Schedule the command completion check
            delay = self.get_step_delay(step_type)
            self.command_timer.start(delay)

            # Start progress bar animation


            # If running a single step, we'll stop after command completion
            if run_single:
                self.single_step_mode = True
            else:
                self.single_step_mode = False

        except Exception as e:
            self.show_error(f"Error executing step: {e}")
            self.stop_program()

    def find_matching_loop_end(self, start_index):
        """Find the matching Loop End for a Loop Start"""
        nesting_level = 0

        for i in range(start_index + 1, len(self.program_steps)):
            step = self.program_steps[i]

            if step.get("type") == "loop_start":
                nesting_level += 1
            elif step.get("type") == "loop_end":
                if nesting_level == 0:
                    return i
                nesting_level -= 1

        return -1

    def execute_vacuum_on_step(self):
        """Activate vacuum (valve on)"""
        if not self.robot:
            raise Exception("Robot not connected")
        try:
            self.robot.SetValveState(0, 1)  # (valve 0: stay, valve 1: open)
            self.log_to_console("Vacuum ON (SetValveState(0,1))")
        except Exception as e:
            self.log_to_console(f"Vacuum ON error: {e}")
            self.show_error("Error during vacuum ON: " + str(e))

    def execute_vacuum_off_step(self):
        """Deactivate vacuum (valve off)"""
        if not self.robot:
            raise Exception("Robot not connected")
        try:
            self.robot.SetValveState(0, 0)  # (valve 0: stay, valve 1: close)
            self.log_to_console("Vacuum OFF (SetValveState(0,0))")
        except Exception as e:
            self.log_to_console(f"Vacuum OFF error: {e}")
            self.show_error("Error during vacuum OFF: " + str(e))
    def check_command_completion(self):
        """Check if the current command has completed"""
        if not self.running:
            return

        # Mark command as complete
        self.command_complete = True

        # Get the current step
        step = self.program_steps[self.current_step_index]

        # Update status
        self.status_label.setText(f"Completed step: {self.get_step_description(step)}")

        # If in single step mode, stop after this step
        if self.single_step_mode:
            self.stop_program()
        else:
            # Continue to next step immediately
            self.execute_next_step()

    def get_step_delay(self, step_type):
        """Get the delay time for a step type"""
        # Different step types need different delays to ensure completion
        if step_type == "move":
            return 2000  # 2 seconds for movement
        elif step_type in ["open_gripper", "close_gripper", "set_gripper"]:
            return 1000  # 1 second for gripper operations
        elif step_type == "home":
            return 3000  # 3 seconds for homing
        elif step_type in ["open_gripper", "close_gripper", "set_gripper", "vacuum_on", "vacuum_off"]:
            return 1000
        else:
            return 500   # 0.5 seconds for other steps

    def friendly_robot_error_message(self, e):
        err_msg = str(e).lower()
        if "singularity" in err_msg:
            return "❗ Singularity error: Linear move not possible due to a singularity along the path."
        elif "not homed" in err_msg or "mx_st_not_homed" in err_msg:
            return "❗ Robot is not homed. Please home the robot."
        elif "limit" in err_msg:
            return f"❗ Limit error: {e}"
        elif "out of reach" in err_msg:
            return "❗ Out of reach: The requested move cannot be performed from the current position."
        elif "connection" in err_msg or "socket" in err_msg:
            return "❗ Connection error: Lost connection to robot."
        else:
            return f"❗ Movement error: {e}"

    def execute_move_step(self, step):
        if not self.robot:
            raise Exception("Robot not connected")

        move_type = step.get("move_type", "MoveL")
        try:
            if move_type == "MoveJ":
                joints = step.get("joints", [0, 0, 0, 0, 0, 0])
                self.robot.MoveJoints(*joints)
                self.robot.WaitIdle()
                self.log_to_console(f"Executed MoveJ to joints {joints}")
            elif move_type == "MoveL":
                position = step.get("position", [180, 0, 180, 0, 0, 0])
                self.robot.MoveLin(*position)
                self.robot.WaitIdle()
                self.log_to_console(f"Executed MoveL to position {position}")
            elif move_type == "MoveP":
                position = step.get("position", [180, 0, 180, 0, 0, 0])
                self.robot.MovePose(*position)
                self.robot.WaitIdle()
                self.log_to_console(f"Executed MoveP to position {position}")
        except Exception as e:
            user_msg = self.friendly_robot_error_message(e)
            self.log_to_console(user_msg)
            self.show_error(user_msg)

    def execute_open_gripper_step(self):
        """Execute an open gripper step"""
        if not self.robot:
            raise Exception("Robot not connected")

        try:
            self.robot.SendCustomCommand("GripperOpen")
            self.gripper_state = "open"
            self.update_gripper_state_label()

            def delayed_update():
                if hasattr(self.robot, "update_gripper_slider"):
                    self.robot.update_gripper_slider()
                self.log_to_console("Gripper opened successfully")

            QTimer.singleShot(500, delayed_update)
            self.log_to_console("Opening gripper...")

        except Exception as e:
            self.log_to_console(f"Gripper error: {e}")
            raise

    def execute_close_gripper_step(self):
        """Execute a close gripper step"""
        if not self.robot:
            raise Exception("Robot not connected")

        try:
            self.robot.SendCustomCommand("GripperClose")
            self.gripper_state = "closed"
            self.update_gripper_state_label()

            # Use QTimer for non-blocking delay
            def delayed_update():
                if hasattr(self.robot, "update_gripper_slider"):
                    self.robot.update_gripper_slider()
                self.log_to_console("Gripper closed successfully")

            QTimer.singleShot(500, delayed_update)
            self.log_to_console("Closing gripper...")

        except Exception as e:
            self.log_to_console(f"Gripper error: {e}")
            raise

    def execute_set_gripper_step(self, step):
        """Execute a set gripper position step"""
        if not self.robot:
            raise Exception("Robot not connected")

        position = step.get("position", 50)
        mm_opening = round((position / 100.0) * 5.8, 2)  # Convert percentage to mm (0-5.8mm)

        try:
            self.robot.MoveGripper(mm_opening)
            self.log_to_console(f"Setting gripper to {position}% ({mm_opening}mm)")

            # Update gripper state based on position
            if position <= 10:
                self.gripper_state = "closed"
            elif position >= 90:
                self.gripper_state = "open"
            else:
                self.gripper_state = "partial"

            self.update_gripper_state_label()
        except Exception as e:
            self.log_to_console(f"Gripper error: {e}")
            raise

    def execute_wait_step(self, step):
        """Execute a wait step"""
        wait_time = step.get("time", 1.0)

        # Convert to milliseconds
        wait_ms = int(wait_time * 1000)

        # Start the wait timer
        self.status_label.setText(f"Waiting for {wait_time} seconds...")
        self.log_to_console(f"Waiting for {wait_time} seconds")
        self.wait_timer.start(wait_ms)

    def wait_completed(self):
        """Called when a wait step is completed"""
        if not self.running:
            return

        self.status_label.setText("Wait completed")

        # Mark command as complete
        self.command_complete = True

        # Continue to the next step immediately
        self.execute_next_step()

    def execute_home_step(self):
        """Execute a home step"""
        if not self.robot:
            raise Exception("Robot not connected")

        try:
            self.robot.MoveJoints(0, 0, 0, 0, 0, 0)
            self.log_to_console("Moving to home position")
        except Exception as e:
            self.log_to_console(f"Home movement error: {e}")
            raise

    def stop_program(self):
        """Stop program execution"""
        self.running = False
        self.current_step_index = -1
        self.loop_stack = []
        self.single_step_mode = False
        self.command_complete = True

        # Stop and disconnect timers
        for t in [self.timer, self.wait_timer, self.command_timer]:
            t.stop()
            try:
                t.timeout.disconnect()
            except TypeError:
                pass  # Already disconnected

        self.status_label.setText("Program stopped")
        self.update_ui_state()


    def program_completed(self):
        """Called when the program is completed"""
        self.running = False
        self.current_step_index = -1
        self.loop_stack = []
        self.single_step_mode = False
        self.command_complete = True

        for t in [self.timer, self.wait_timer, self.command_timer]:
            t.stop()
            try:
                t.timeout.disconnect()
            except TypeError:
                pass  # Already disconnected

        self.status_label.setText("Program completed")
        self.log_to_console("Program execution completed successfully")
        self.update_ui_state()

        # Update status after a delay

    def update_steps_list(self):
        """Update the steps list widget"""
        self.steps_list.clear()

        for i, step in enumerate(self.program_steps):
            item_text = f"{i+1}. {self.get_step_description(step)}"
            self.steps_list.addItem(item_text)

        # Enable/disable Run Program button based on whether there are steps
        self.run_btn.setEnabled(len(self.program_steps) > 0)

    def get_step_description(self, step):
        """Get a human-readable description of a step"""
        step_type = step.get("type", "")

        if step_type == "move":
            move_type = step.get("move_type", "MoveL")
            speed = step.get("speed", 20.0)
            # Show different coordinates based on move type
            if move_type == "MoveJ":
                joints = step.get("joints", [0, 0, 0, 0, 0, 0])
                joints_str = ", ".join(f"{j:.1f}" for j in joints)
                return f"{move_type} to joints [{joints_str}] @ {speed}%"
            else:
                position = step.get("position", [0, 0, 0, 0, 0, 0])
                pos_str = ", ".join(f"{p:.1f}" for p in position[:3])
                return f"{move_type} to position [{pos_str}...] @ {speed}%"

        elif step_type == "open_gripper":
            return "Open Gripper"

        elif step_type == "close_gripper":
            return "Close Gripper"

        elif step_type == "set_gripper":
            position = step.get("position", 0)
            return f"Set Gripper to {position}%"

        elif step_type == "wait":
            time = step.get("time", 1.0)
            return f"Wait {time} seconds"

        elif step_type == "home":
            return "Move to Home Position"

        elif step_type == "loop_start":
            count = step.get("count", 1)
            return f"Loop Start (repeat {count} times)"

        elif step_type == "loop_end":
            return "Loop End"

        return "Unknown Step"

    def execute_true_home(self):
        """Execute the robot's true homing condition using the Home command"""
        if not self.robot:
            self.show_error("Robot not connected")
            return

        if self.running:
            self.show_error("Cannot home while program is running")
            return

        # First ensure we're synced with robot state
        self.sync_robot_state()

        try:
            # Use a safer approach - always reset first
            self.log_to_console("Starting safe homing sequence...")

            # Step 1: Reset any errors
            try:
                self.robot.ResetError()
                self.log_to_console("Reset any errors")
                time.sleep(1.0)  # Give robot time to process
            except Exception as reset_err:
                self.log_to_console(f"Reset warning (continuing): {reset_err}")
                # Continue anyway - error might not exist

            # Step 2: Deactivate robot if needed
            try:
                self.robot.DeactivateRobot()
                self.log_to_console("Robot deactivated")
                time.sleep(1.0)  # Give robot time to process
            except Exception as deact_err:
                self.log_to_console(f"Deactivation warning (continuing): {deact_err}")
                # Continue anyway - might already be deactivated

            # Step 3: Activate robot
            try:
                self.robot.ActivateRobot()
                self.log_to_console("Robot activated")
                time.sleep(1.0)  # Give robot time to process
            except Exception as act_err:
                self.log_to_console(f"Activation error: {act_err}")
                self.show_error(f"Could not activate robot: {str(act_err)}")
                return

            # Step 4: Home robot
            try:
                # Set status and execute Home command
                self.status_label.setText("Homing robot...")
                self.log_to_console("Executing Home command")

                # Execute the Home command
                self.robot.Home()

                # Wait for homing to complete
                self.log_to_console("Waiting for homing to complete...")
                self.robot.WaitHomed()

                # Success!
                self.log_to_console("Homing completed successfully")
                self.status_label.setText("Robot homed successfully")

            except Exception as home_err:
                self.log_to_console(f"Homing error: {home_err}")
                self.show_error(f"Error during homing: {str(home_err)}")

        except Exception as e:
            self.log_to_console(f"Unexpected error: {e}")
            self.show_error(f"Unexpected error: {str(e)}")

    def update_ui_state(self):
        """Update the UI state based on program execution state"""
        running = self.running

        # Enable/disable buttons based on program state
        self.add_step_btn.setEnabled(not running)
        self.edit_step_btn.setEnabled(not running)
        self.delete_step_btn.setEnabled(not running)
        self.move_up_btn.setEnabled(not running)
        self.move_down_btn.setEnabled(not running)
        self.copy_step_btn.setEnabled(not running)
        self.paste_step_btn.setEnabled(not running)
        self.record_pos_btn.setEnabled(not running)
        self.true_home_btn.setEnabled(not running)
        self.load_btn.setEnabled(not running)
        self.save_btn.setEnabled(not running)
        self.run_btn.setEnabled(not running and len(self.program_steps) > 0)
        self.run_step_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def update_gripper_state_label(self):
        """Update the gripper state label"""
        if not hasattr(self, 'gripper_state_label'):
            return

        if self.gripper_state == "open":
            self.gripper_state_label.setStyleSheet("color: green;")
        elif self.gripper_state == "closed":
            self.gripper_state_label.setStyleSheet("color: red;")
        elif self.gripper_state == "partial":
            self.gripper_state_label.setStyleSheet("color: orange;")
        else:
            self.gripper_state_label.setStyleSheet("color: gray;")

    def show_error(self, message):
        if getattr(self, '_last_error_message', None) == message:
            return
        self._last_error_message = message
        self.log_to_console(f"ERROR: {message}")

        # Update error display
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        self.reset_error_btn.setVisible(True)  # It's often better to show the reset button

        if self.running:
            self.stop_program()

        # Use a timer to hide the error message automatically
        self.error_timer.start(5000)
    def reset_error(self):
        """Reset the error state"""
        # Disable the reset button immediately to prevent double-clicks
        self.reset_error_btn.setEnabled(False)

        # Update UI state
        self.error_label.setVisible(False)
        self.reset_error_btn.setVisible(False)


        # Show status message
        self.status_label.setText("Resetting robot error...")

        # Try to reset robot error if connected
        if self.robot:
            try:
                # Wrap everything in try-except to prevent crashes
                try:
                    # Check if robot is still connected
                    try:
                        # Use a simple command that won't cause issues if robot is in error state
                        status = self.robot.GetStatusRobot()
                        self.log_to_console(f"Robot status before reset: {status}")
                    except Exception as status_err:
                        # If we can't get status, try to reconnect first
                        self.log_to_console(f"Cannot get robot status: {status_err}, attempting to reconnect...")
                        try:
                            # Try to disconnect cleanly first
                            try:
                                self.robot.Disconnect()
                            except:
                                pass

                            time.sleep(1.5)  # Wait longer for socket to fully close

                            # Reconnect
                            self.robot.Connect(address, disconnect_on_exception=False)
                            self.log_to_console("Reconnected to robot")
                            time.sleep(0.5)  # Give connection time to stabilize
                        except Exception as conn_err:
                            self.log_to_console(f"Reconnection failed: {conn_err}")
                            # Continue anyway - we'll try to reset

                    # Step 1: Reset any errors
                    self.log_to_console("Attempting to reset robot error...")
                    self.robot.ResetError()
                    time.sleep(1.0)  # Give robot time to process
                    self.log_to_console("Robot error reset command sent")

                    # Step 2: Deactivate robot if needed
                    try:
                        self.log_to_console("Deactivating robot...")
                        self.robot.DeactivateRobot()
                        time.sleep(1.0)  # Give robot time to process
                        self.log_to_console("Robot deactivated")
                    except Exception as deact_err:
                        self.log_to_console(f"Deactivation warning (continuing): {deact_err}")
                        # Continue anyway - might already be deactivated

                    # Step 3: Activate robot
                    try:
                        self.log_to_console("Activating robot...")
                        self.robot.ActivateRobot()
                        time.sleep(1.0)  # Give robot time to process
                        self.log_to_console("Robot activated successfully")
                    except Exception as act_err:
                        self.log_to_console(f"Could not activate robot: {act_err}")
                        # Don't raise, just log the error

                    # Update status
                    self.status_label.setText("Robot error reset complete")

                except Exception as inner_err:
                    self.log_to_console(f"Error during reset sequence: {inner_err}")
                    self.status_label.setText("Error reset failed")
                    # Don't re-raise, contain the error

            except Exception as e:
                # This should never happen due to inner try-except, but just in case
                self.log_to_console(f"Unexpected error during reset: {e}")
                self.status_label.setText("Error reset failed")

        # Re-enable the reset button after a delay to prevent accidental double-clicks

    def sync_robot_state(self):
        if not self.robot:
            return
        if self.running:
            return

        try:
            status = self.robot.GetStatusRobot()
            # Update cached state, but do NOT log anything
            self._prev_robot_status = str(status)
            self._prev_activated = getattr(status, 'activation_state', None)
            self._prev_homed = getattr(status, 'homed', None)
            self._prev_error = getattr(status, 'error', None)
            # (You may use these for GUI indicators elsewhere, but do NOT log here.)
        except Exception:
            # Optionally handle exceptions, but do NOT log or print
            pass
    def log_to_console(self, message):
        """Log a message to the console"""
        # Emit signal to update console in main GUI
        self.console_update.emit(message)


def add_programming_interface_to_gui(main_gui):
    """
    Add the programming interface to the main GUI below the console.

    Args:
        main_gui: The main GUI window instance

    Returns:
        The created programming interface instance
    """
    # Create the programming interface
    programming_interface = ProgrammingInterface(robot=main_gui.robot)

    # Connect the console update signal to the main GUI's log method
    programming_interface.console_update.connect(main_gui.log)

    return programming_interface


# Add a help dialog for movement types
class MovementTypeHelpDialog(QDialog):
    """Dialog showing help information about movement types"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Movement Types Help")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Add title
        title = QLabel("Robot Movement Types")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Add description
        description = QLabel(
            "The Meca500 robot supports three types of movements, each with different characteristics "
            "and suitable for different situations. Understanding when to use each type will help you "
            "create more efficient and reliable robot programs."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Add movement type descriptions
        for move_type, desc in MOVEMENT_TYPE_DESCRIPTIONS.items():
            group = QGroupBox(move_type)
            group_layout = QVBoxLayout()
            group.setLayout(group_layout)

            label = QLabel(desc)
            label.setWordWrap(True)
            group_layout.addWidget(label)

            # Add specific details
            if move_type == "MoveJ":
                details = QLabel(
                    "• Fastest movement type\n"
                    "• Each joint moves independently\n"
                    "• Path is not predictable in Cartesian space\n"
                    "• Best for free movements when path doesn't matter\n"
                    "• Good for avoiding singularities"
                )
            elif move_type == "MoveL":
                details = QLabel(
                    "• Robot follows a straight line in Cartesian space\n"
                    "• Predictable path\n"
                    "• Slower than MoveJ\n"
                    "• Best for precise movements around workpieces\n"
                    "• May encounter singularities"
                )
            else:  # MoveP
                details = QLabel(
                    "• Similar to MoveL but with smoother acceleration\n"
                    "• Best for continuous paths with multiple points\n"
                    "• Good for drawing or following contours\n"
                    "• May encounter singularities\n"
                    "• Slightly slower than MoveL"
                )

            group_layout.addWidget(details)
            layout.addWidget(group)

        # Add recommendation
        recommendation = QLabel(
            "<b>Recommendation:</b> Start with MoveJ for general movements, use MoveL when you need "
            "precise straight-line movements, and MoveP for continuous paths with multiple points."
        )
        recommendation.setWordWrap(True)
        layout.addWidget(recommendation)

        # Add close button clos
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


# Function to show movement type help
def show_movement_type_help(parent=None):

    dialog = MovementTypeHelpDialog(parent)
    dialog.exec()