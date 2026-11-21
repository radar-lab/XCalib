"""Run a folder dialog (or, with --files, a file dialog) in its own Qt process, outside the notebook kernel."""
import argparse
import json


def main():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFileDialog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title', required=True)
    parser.add_argument('--initial', default='')
    parser.add_argument('--files', default='',
                        help='pick one existing file with this name filter, e.g. "Matrix (*.json *.txt)"')
    args = parser.parse_args()
    app = QApplication([])
    dialog = QFileDialog(None, args.title, args.initial)
    if args.files:
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(args.files)
    else:
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)
    dialog.setOption(QFileDialog.DontUseNativeDialog)
    dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
    selected = dialog.selectedFiles()[0] if dialog.exec() else ''
    print(json.dumps(selected), flush=True)
    app.quit()


if __name__ == '__main__':
    main()
