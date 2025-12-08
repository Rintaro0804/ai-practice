from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # 再度インポート（念のため）


# データセットを用意
digits = datasets.load_digits()

# 画像データと正解ラベルを用意
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 学習用データとテスト用データを用意
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# 学習
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

# 予測
predicted = clf.predict(X_test)

# 正解率を表示
print(f"正解率 (Accuracy): {metrics.accuracy_score(y_test, predicted)}")

# 予測結果と正解結果を表示
print("予測結果")
print(predicted[:10])

print("正解結果")
print(y_test[:10])

print("\nどこで間違えたかの内訳（混同行列）")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
plt.show()