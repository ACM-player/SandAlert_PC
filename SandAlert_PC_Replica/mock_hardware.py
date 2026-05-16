"""
SandAlert PC - 零污染虚拟串口硬件模拟器
=========================================
功能：在 macOS 上创建一个虚拟串口（伪终端），
从指定的历史数据文件夹中读取所有 .txt 文件，
逐行向虚拟串口发送数据，模拟真实 STM32 硬件的行为。

使用方法：
  1. 先运行本模拟器，它会打印出虚拟串口名称（如 /dev/ttys001）
  2. 再启动 SandAlert PC 软件，在 Setup -> Comm Settings... 中
     选择对应的串口名称和波特率 19200，点击连接
  3. 回到本模拟器终端，按回车开始发送数据

注意事项：
  - 本模拟器使用 macOS 内置的 pty 伪终端，无需任何硬件
  - 发送格式：每行一个浮点数（电压值），与真实硬件保持一致
  - 所有 .txt 文件会按文件名排序后循环发送
"""

import os
import pty
import time
import glob

def start_mock_hardware(data_dir="."):
    """
    启动虚拟串口硬件模拟器
    
    参数:
        data_dir: 包含 .txt 数据文件的文件夹路径，默认为当前目录
    """
    # 扫描文件夹下所有 .txt 文件
    search_pattern = os.path.join(data_dir, "*.txt")
    txt_files = sorted(glob.glob(search_pattern))
    
    if not txt_files:
        print(f"❌ 在 {data_dir} 中未找到任何 .txt 文件！")
        print(f"   请确保数据文件存放在正确的路径下。")
        return
    
    print(f"📂 找到 {len(txt_files)} 个数据文件:")
    for f in txt_files:
        print(f"   - {os.path.basename(f)}")
    
    # 创建伪终端（虚拟串口）
    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)
    
    print("=" * 60)
    print("🚀 [零污染硬件模拟器] 已启动！")
    print(f"👉 请在 SandAlert PC 的 Setup -> Comm Settings... 中：")
    print(f"   Com Port 设置为: {slave_name}")
    print(f"   Baud Rate 设置为: 19200")
    print("=" * 60)
    
    # 等待用户在 SandAlert PC 中完成串口连接设置
    input("\n[等待操作] 请在软件中连接上述串口后，按回车键开始发送数据...")
    
    print(f"\n📡 开始向 {slave_name} 循环发送数据（共 {len(txt_files)} 个文件）...")
    print("   按 Ctrl+C 可安全停止模拟器。\n")
    
    try:
        file_index = 0
        while True:
            # 轮询发送所有文件，形成无限循环
            file_path = txt_files[file_index % len(txt_files)]
            file_name = os.path.basename(file_path)
            print(f"📄 正在发送: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 终极修复：直接使用无参数的 split()，它会自动完美处理所有空格和 Tab
                    tokens = line.split()
                    
                    for token in tokens:
                        try:
                            val = float(token)
                            # 每行发送一个浮点数，模拟真实串口行为
                            os.write(master, (f'{val}\n').encode('utf-8'))
                            time.sleep(0.001)
                        except ValueError:
                            # 遇到非数字（比如文件头的说明文字）直接静默跳过
                            pass
            
            file_index += 1
            print(f"✅ {file_name} 发送完成，准备下一个文件...")
    
    except KeyboardInterrupt:
        print("\n🛑 模拟器已安全关闭。")
    finally:
        os.close(master)
        os.close(slave)

if __name__ == "__main__":
    import sys
    
    # ====== 默认数据路径：外置盘上的现场数据文件夹 ======
    DEFAULT_DATA_DIR = "/Volumes/LKS/中海油能源发展装备公司海上油气管道含砂监测技术应用研究/结题资料/现场数据"
    
    # 如果命令行传入了文件夹路径，则使用它
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = DEFAULT_DATA_DIR
        print(f"📌 未指定数据路径，使用默认路径：{data_dir}")
        print(f"   如需自定义路径，请运行: python mock_hardware.py /你的/数据/路径\n")
    
    # 检查路径是否存在
    if not os.path.isdir(data_dir):
        print(f"❌ 路径不存在: {data_dir}")
        print(f"   请检查外置盘是否已挂载，或手动传入正确的数据路径。")
        sys.exit(1)
    
    start_mock_hardware(data_dir)
