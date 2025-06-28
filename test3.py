import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

data_path = "gossipcop_v5_tiny_balanced.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取10条新闻文本和标签
texts = []
labels = []
for item in list(data.values())[:10]:
    text = item["text"]
    label = 0 if item["label"] == "fake" else 1  # 0 for fake, 1 for true
    texts.append(text)
    labels.append(label)

# 数据预处理
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

processed_texts = [preprocess(text) for text in texts]

# 构建词典和语料库
dictionary = Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# 训练LDA模型
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 获取主题分布
doc_topic_probs = []
for doc in corpus:
    topic_dist = lda_model[doc]
    prob_dist = [0.0] * num_topics
    for topic, prob in topic_dist:
        prob_dist[topic] = prob
    doc_topic_probs.append(prob_dist)

# 获取情感分析结果
def analyze_sentiment(text):
    prompt = f"请分析以下文本的情感倾向，判断是积极、消极还是中性。请务必从以下三种情感倾向中选择一种进行回答：\n\n文本：\n{text}\n\n情感倾向："
    response = call_model(prompt)
    print(f"情感分析模型返回内容：{response}")
    match = re.search(r"情感倾向：(积极|消极|中性)", response)
    if match:
        return match.group(1)
    else:
        return "中性"


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


sentiment_results = []
for text in texts:
    sentiment = analyze_sentiment(text)
    sentiment_results.append(sentiment)

# 将情感倾向转换为数值分数
sentiment_scores = []
for sentiment in sentiment_results:
    if sentiment == "积极":
        sentiment_scores.append(1.0)
    elif sentiment == "消极":
        sentiment_scores.append(-1.0)
    else:
        sentiment_scores.append(0.0)

# 提取主题关键词
topic_words_list = []
for topic_idx in range(num_topics):
    topic_words = lda_model.show_topic(topic_idx, topn=10)
    topic_words_list.append([word for word, prob in topic_words])

# 将主题关键词列表转换为字符串
topic_words_str = [", ".join(words) for words in topic_words_list]

# 特征融合
features = []
for i in range(len(doc_topic_probs)):
    feature = doc_topic_probs[i] + [sentiment_scores[i]]
    features.append(feature)

# 标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 构建神经网络模型
inputs = Input(shape=(features.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 将标签转换为独热编码
labels_categorical = to_categorical(labels)

# 训练模型
history = model.fit(features, labels_categorical, epochs=50, batch_size=2, verbose=1, validation_split=0.2)

# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 预测
predictions = model.predict(features)
predicted_labels = np.argmax(predictions, axis=1)

# 计算准确率
accuracy = np.mean(predicted_labels == labels)
print(f"模型准确率: {accuracy:.2f}")

# 综合分析
for i in range(len(texts)):
    text = texts[i]
    label = labels[i]
    doc_topic = doc_topic_probs[i]
    sentiment = sentiment_results[i]
    predicted_label = predicted_labels[i]

    prompt = f"请结合以下信息分析新闻的真实性：\n\n新闻文本：\n{text}\n\n主题分布：新闻涉及的主题包括 {', '.join(topic_words_str)}\n\n情感倾向：新闻的情感倾向为 {sentiment}\n\n预测结果：{'真' if predicted_label == 1 else '假'}\n\n综合分析："
    analysis = call_model(prompt)
    print(f"\n新闻ID: gossipcop-{i + 1}")
    print(f"实际标签: {'真' if label == 1 else '假'}")
    print(f"预测结果: {'真' if predicted_label == 1 else '假'}")
    print(f"综合分析: {analysis}")