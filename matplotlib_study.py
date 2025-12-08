import matplotlib.pyplot as plt 
import numpy as np

# 1. データの準備 (NumPyを使います)
# 0から10までを、100分割した数字の列を作る（時間の流れをシミュレーション）
x = np.linspace(0, 10, 100)

y = np.sin(x)

plt.plot(x, y, label = "Steering Angle", color="blue")

plt.title("Autonomous Driving Simulation") # グラフのタイトル
plt.xlabel("Time (sec)") # x軸のラベル
plt.ylabel("Angle") # y軸のラベル
plt.grid(True) # グリッドを表示
plt.legend() # 凡例を表示
plt.show()