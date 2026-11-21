"""Keyboard frame navigation of the review window."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QLineEdit, QWidget


class FrameNavigation(QObject):
    """Intercept frame arrows within a window, except in editing controls."""

    def __init__(self, window, step):
        super().__init__(window)
        self.window = window
        self.step = step
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        if (event.type() != QEvent.KeyPress or not isinstance(obj, QWidget)
                or obj.window() is not self.window or event.modifiers() != Qt.NoModifier
                or event.key() not in (Qt.Key_Left, Qt.Key_Right)):
            return False
        # Qt can propagate an unhandled key from an editor to its parent.
        # Check the focus widget as well as the event receiver so the parent
        # does not accidentally turn that same arrow into frame navigation.
        for widget in (obj, self.window.focusWidget()):
            while widget is not None and widget is not self.window:
                if isinstance(widget, (QLineEdit, QAbstractSpinBox, QComboBox)):
                    return False
                widget = widget.parentWidget()
        self.step(-1 if event.key() == Qt.Key_Left else 1)
        event.accept()
        return True
