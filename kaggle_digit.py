import pandas as pd
from sklearn import svm

# データを読み込む
print("loading data...")

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# データを学習用とテスト用に分ける
y_train = train_data["label"]
X_train = train_data.drop("label", axis=1)

X_test = test_data

# モデルを訓練する
print("training model...")
clf = svm.SVC()
clf.fit(X_train, y_train)

# 予測する
print("predicting...")
predictions = clf.predict(X_test)

# 提出用のデータフレームを作る
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

# 提出用のCSVファイルを出力する
submission.to_csv("submission.csv", index=False)
print("submission.csvを出力しました")