import json
import matplotlib.pyplot as plt

# ========== 请改成你的JSON实际路径 ==========
json_path = "predict_results2.json"
# ==========================================

with open(json_path, "r", encoding="utf-8") as f:
    results = json.load(f)

audio_ai = 0
audio_total = 0
other_real = 0
other_total = 0
mid_prob = 0
total = len(results)

for item in results:
    fn = item["文件名"]
    pred = item["判定结果"]
    prob = item["AI合成概率"]

    if fn.startswith("audio"):
        audio_total += 1
        if pred == "AI合成声":
            audio_ai += 1
    else:
        other_total += 1
        if pred == "真实人声":
            other_real += 1

    if 0.2 <= prob <= 0.8:
        mid_prob += 1

p_audio_ai = audio_ai / audio_total * 100 if audio_total else 0
p_other_real = other_real / other_total * 100 if other_total else 0
p_mid = mid_prob / total * 100 if total else 0

print(f"audio开头判为AI比例：{p_audio_ai:.2f}%")
print(f"其他开头判为人声比例：{p_other_real:.2f}%")
print(f"概率0.3~0.7区间比例：{p_mid:.2f}%")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["audio开头→AI", "其他开头→人声", "概率0.3~0.7"]
values = [p_audio_ai, p_other_real, p_mid]
colors = ["#4472C4", "#ED7D31", "#70AD47"]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, values, color=colors)
plt.ylabel("占比 (%)")
plt.title("模型预测结果统计")
plt.ylim(0, 105)

for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.show()