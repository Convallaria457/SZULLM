import requests
import json
import re

# 加载数据
data_path = "gossipcop_v5_tiny_balanced.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 模型调用函数
def call_model(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1:8b",
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.1,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("response", "无法解析模型返回结果")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return "请求失败"


# 真伪判断函数
def detect_fake_news(text, image_description):
    prompt = f"请分析以下新闻文本和图片描述，明确判断这条新闻是真还是假。如果新闻是真实的，请回答'判断结果：真'，如果是假的，请回答'判断结果：假'：\n\n新闻文本：\n{text}\n\n图片描述：\n{image_description}\n\n判断结果："
    response = call_model(prompt)
    # 打印模型返回的完整内容
    print(f"模型返回内容：{response}")
    # 使用正则表达式提取判断结果
    match = re.search(r"判断结果：(真|假)", response)
    if match:
        return match.group(1)
    else:
        return "不确定"


# 情感分析函数
def analyze_sentiment(text):
    prompt = f"请分析以下文本的情感倾向，判断是积极、消极还是中性。请务必从以下三种情感倾向中选择一种进行回答：\n\n文本：\n{text}\n\n情感倾向："
    response = call_model(prompt)
    # 使用正则表达式提取判断结果
    match = re.search(r"情感倾向：(积极|消极|中性)", response)
    if match:
        return match.group(1)
    else:
        return "中性"


# 结合情感分析的真伪判断函数
def detect_fake_news_with_sentiment(text, image_description, sentiment):
    prompt = f"请分析以下文本新闻、图片描述和情感倾向，明确判断这条新闻是真还是假。如果新闻是真实的，请回答'判断结果：真'，如果是假的，请回答'判断结果：假'：\n\n新闻文本：\n{text}\n\n图片描述：\n{image_description}\n\n情感倾向：\n{sentiment}\n\n判断结果："
    response = call_model(prompt)
    print(f"模型返回内容：{response}")
    match = re.search(r"判断结果：(真|假)", response)
    if match:
        return match.group(1)
    else:
        return "不确定"


# 统计准确率
total = len(data)
correct = 0
correct_fake = 0
total_fake = 0
correct_true = 0
total_true = 0

#判断真伪统计准确率
print("逐条处理结果：")
for item in data.values():
    label = item["label"]
    text = item["text"]
    image_description = item.get("generated_image_description_glm4", "")

    prediction = detect_fake_news(text, image_description)

    if label == "legitimate":
        total_true += 1
        result = "正确" if prediction == "真" else "错误"
        if prediction == "真":
            correct += 1
            correct_true += 1
    else:
        total_fake += 1
        result = "正确" if prediction == "假" else "错误"
        if prediction == "假":
            correct += 1
            correct_fake += 1
    print(f"新闻ID: {item['id']}, 判断结果: {prediction}, 实际标签: {label}, 判断是否正确: {result}")

accuracy = correct / total if total > 0 else 0
accuracy_fake = correct_fake / total_fake if total_fake > 0 else 0
accuracy_true = correct_true / total_true if total_true > 0 else 0

print("\n第一步：判断真伪统计准确率")
print(f"总体准确率: {accuracy:.2f}")
print(f"假新闻准确率: {accuracy_fake:.2f}")
print(f"真新闻准确率: {accuracy_true:.2f}")

#情感分析
print("\n第二步：情感分析结果")
for item in data.values():
    text = item["text"]
    sentiment = analyze_sentiment(text)
    print(f"新闻ID: {item['id']}, 情感: {sentiment}")

#结合情感分析判断真伪统计准确率
correct_with_sentiment = 0
correct_fake_with_sentiment = 0
correct_true_with_sentiment = 0

print("\n第三步：结合情感分析判断真伪逐条结果")
for item in data.values():
    label = item["label"]
    text = item["text"]
    image_description = item.get("generated_image_description_glm4", "")
    sentiment = analyze_sentiment(text)

    prediction = detect_fake_news_with_sentiment(text, image_description, sentiment)

    if label == "legitimate":
        result = "正确" if prediction == "真" else "错误"
        if prediction == "真":
            correct_with_sentiment += 1
            correct_true_with_sentiment += 1
    else:
        result = "正确" if prediction == "假" else "错误"
        if prediction == "假":
            correct_with_sentiment += 1
            correct_fake_with_sentiment += 1

    print(f"新闻ID: {item['id']}, 判断结果: {prediction}, 实际标签: {label}, 判断是否正确: {result}")

# 计算准确率
accuracy_with_sentiment = correct_with_sentiment / total if total > 0 else 0
accuracy_fake_with_sentiment = correct_fake_with_sentiment / total_fake if total_fake > 0 else 0
accuracy_true_with_sentiment = correct_true_with_sentiment / total_true if total_true > 0 else 0

print("\n第三步：结合情感分析判断真伪统计准确率")
print(f"总体准确率: {accuracy_with_sentiment:.2f}")
print(f"假新闻准确率: {accuracy_fake_with_sentiment:.2f}")
print(f"真新闻准确率: {accuracy_true_with_sentiment:.2f}")

# 比较准确率提升
print("\n准确率提升对比：")
print(f"总体准确率提升: {accuracy_with_sentiment - accuracy:.2f}")
print(f"假新闻准确率提升: {accuracy_fake_with_sentiment - accuracy_fake:.2f}")
print(f"真新闻准确率提升: {accuracy_true_with_sentiment - accuracy_true:.2f}")