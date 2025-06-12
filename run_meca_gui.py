"""
Meca500 GUI
---------------------------------------
Entry point for the Meca500 GUI.

This module serves as the main entry point for the Meca500 robot control application.
"""
import sys
import traceback
from PyQt6.QtWidgets import QApplication
from improved_error_handling import patch_meca_pendant
from Test import MecaPendant


def main():
    """
    Main application entry point.
    Initializes the GUI application with improved error handling.
    """
    try:
        app = QApplication(sys.argv)

        # Apply error handling improvements to the MecaPendant class
        ImprovedMecaPendant = patch_meca_pendant(MecaPendant)

        # Create and show the main window
        window = ImprovedMecaPendant()
        window.show()

        # Start the application event loop
        sys.exit(app.exec())

    except Exception as e:
        # Catch any unhandled exceptions to prevent silent crashes
        print(f"Fatal error in main application: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


