"""
Meca500 GUI with Improved Error Handling
---------------------------------------
Entry point for the Meca500 GUI with improved error handling.

"""
import sys
from PyQt6.QtWidgets import QApplication

# Import the improved error handling
from improved_error_handling import patch_meca_pendant

# Import the original MecaPendant class
from Test import MecaPendant


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Patch the MecaPendant class with improved error handling
    ImprovedMecaPendant = patch_meca_pendant(MecaPendant)

    # Create and show the main window
    window = ImprovedMecaPendant()
    window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
