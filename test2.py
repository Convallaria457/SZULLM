import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import pyLDAvis.gensim_models
import pyLDAvis
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# 加载数据
data_path = "gossipcop_v5_tiny_balanced.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取10条新闻文本
texts = []
for item in list(data.values())[:10]:
    text = item["text"]
    texts.append(text)

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

# pyLDAvis交互图
lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')  # 保存为HTML文件
print("pyLDAvis交互图已保存为 lda_visualization.html")

# 词云图
for i in range(num_topics):
    topic_words = lda_model.show_topic(i, topn=20)
    topic_words_dict = {word: prob for word, prob in topic_words}
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(topic_words_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Topic {i}')
    plt.axis('off')
    plt.show()

# 热力图
import numpy as np
import seaborn as sns

doc_topic_probs = []
for doc in corpus:
    topic_dist = lda_model[doc]
    prob_dist = [0.0] * num_topics
    for topic, prob in topic_dist:
        prob_dist[topic] = prob
    doc_topic_probs.append(prob_dist)

doc_topic_probs = np.array(doc_topic_probs)
plt.figure(figsize=(12, 8))
sns.heatmap(doc_topic_probs, annot=True, cmap="YlGnBu", xticklabels=[f'Topic {i}' for i in range(num_topics)])
plt.title('Document-Topic Probability Distribution')
plt.xlabel('Topics')
plt.ylabel('Documents')
plt.show()

# 结合大模型分析各主题的内容
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

# 分析每个主题的内容
for i in range(num_topics):
    topic_words = lda_model.show_topic(i, topn=10)
    topic_words_list = [word for word, prob in topic_words]
    topic_prompt = f"以下词汇代表了一个主题：{', '.join(topic_words_list)}。请详细描述这个主题可能涉及的内容："
    topic_analysis = call_model(topic_prompt)
    print(f"\n主题 {i} 分析：")
    print(topic_analysis)