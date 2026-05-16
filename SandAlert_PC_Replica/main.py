# import sys
# from PySide6.QtWidgets import QApplication

# # 引入主窗口
# from ui.main_window import MainWindow

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion") 
    
#     window = MainWindow()
#     window.show()
    
#     sys.exit(app.exec())

import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow

if __name__ == "__main__":
    print(">>> 程序开始启动，正在初始化界面...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    window = MainWindow()
    window.show()
    
    # 修复点 3：Mac 专属唤醒，强制把窗口拉扯到所有软件的最前面
    window.raise_()
    window.activateWindow()
    
    print(">>> 界面加载完成，请点击菜单栏 Setup -> Comm Settings...")
    sys.exit(app.exec())