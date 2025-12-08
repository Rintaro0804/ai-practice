import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets # 練習用データセットが入っている箱

digits = datasets.load_digits()

print("コンピュータが見ている画像データ")
print(digits.images[0])

print("正解ラベル")
print(digits.target[0])

# 人間にわかるよう画像として表示
plt.imshow(digits.images[0], cmap = plt.cm.gray_r)
plt.title(f"Label: {digits.target[0]}")
plt.show()