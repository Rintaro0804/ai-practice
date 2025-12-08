import pandas as pd 

data = {
    "時刻": ["10:00", "10:01", "10:02", "10:03"],
    "速度(km/h)": [40, 45, 60, 30],
    "ブレーキ": [0, 0, 0, 1]  # 0:なし, 1:あり
}

df = pd.DataFrame(data)

print(df)

#平均速度を計算
avg_speed = df["速度(km/h)"].mean()
print("平均速度:", avg_speed)

#ブレーキを押した瞬間を抽出
brake_data = df[df["ブレーキ"] == 1]
print("ブレーキを押した瞬間:", brake_data)
