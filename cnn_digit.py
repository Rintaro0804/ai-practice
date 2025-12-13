import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# データの読み込み
train_data = pd.read_csv("data/train.csv")

# データの前処理
y = train_data["label"]
X = train_data.drop("label", axis=1)

# データの整形
X = X.values.reshape(-1, 28, 28, 1)

# データの正規化
X = X / 255.0

# データの分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# モデルの構築
model = keras.Sequential([
    # 1層目
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),

    # 2層目
    layers.MaxPooling2D(pool_size=(2, 2)),

    # 3層目
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # 平坦化
    layers.Flatten(),

    # 出力層
    layers.Dense(10, activation="softmax")
])

# モデルのコンパイル
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# モデルの訓練
print("start training")
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))


print("Kaggle提出用データの作成")

# Kaggle提出用データの作成
test_data = pd.read_csv("data/test.csv")

# データの整形
X_test = test_data.values.reshape(-1, 28, 28, 1)
X_test = X_test / 255.0

# 予測
predictions_proba = model.predict(X_test)

# 最も確率が高い数字を選ぶ
predictions = np.argmax(predictions_proba, axis=1)

# Kaggle提出用データの作成
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

# Kaggle提出用データの保存
submission.to_csv("cnn_submission.csv", index=False)

print("Kaggle提出用データの作成完了")
