#narrative_clustering.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import numpy as np

print("🚀 正在加载数据并清洗文本...")
# 1. 加载数据
df = pd.read_csv('data/ma_corpus.csv')
df_clean = df.dropna(subset=['public_sentiment_mean', 'public_text']).copy()
df_clean = df_clean[df_clean['public_text_count'] > 0]
df_clean.reset_index(drop=True, inplace=True)

# 2. 停用词与清洗 (保持我们之前优化过的最强版本)
custom_stop_words = [
    'united', 'states', 'securities', 'exchange', 'commission', 'form', 'false',
    'inc', 'corp', 'llc', 'company', 'date', 'report', 'pursuant', 'section',
    'item', 'exhibit', 'signed', 'signature', 'hereinafter', 'thereto', 'hereby',
    'act', 'registrant', 'rule', 'incorporated', 'washington', 'dc', 'edgar',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december', 'file', 'number', 'address',
    'telephone', 'zip', 'code', 'incorporation', 'delaware', 'amendment', 'agreement',
    'merger', 'sub', 'parent', 'stock', 'share', 'shares', 'board', 'directors',
    'common', 'per', 'value', 'par', 'article', 'new', 'york', 'corporation',
    'financial', 'proxy', 'statement', 'information', 'annual', 'quarterly', 'period',
    'ended', 'shall', 'thereto', 'thereof', 'business', 'day', 'days', 'time',
    'including', 'certain', 'respect', 'required'
]
final_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))


def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df_clean['clean_official_text'] = df_clean['official_text'].apply(preprocess_text)

print("📊 正在提取 TF-IDF 特征矩阵...")
# 3. TF-IDF 特征提取
tfidf_vectorizer = TfidfVectorizer(stop_words=final_stop_words, max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['clean_official_text'])

print("🧠 正在训练 K-Means 聚类模型 (设定为 3 个流派)...\n")
# 4. K-Means 聚类
# 假设企业的公关话术主要分为 3 大类
num_clusters = 3
# 使用 random_state 确保每次运行结果一致，方便写入报告
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

# 将聚类结果贴回原始数据
df_clean['cluster'] = kmeans.labels_

# 5. 分析并输出每个“叙事流派”的核心特征与社会反响
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

print("================= 深度建模分析结果 =================")
for i in range(num_clusters):
    print(f"\n🔹 流派 (Cluster) {i}:")

    # 提取该流派最核心的 10 个词汇
    top_words = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"核心修辞词汇: {', '.join(top_words)}")

    # 计算该流派的社会反响 (平均情感得分)
    cluster_df = df_clean[df_clean['cluster'] == i]
    mean_sentiment = cluster_df['public_sentiment_mean'].mean()
    event_count = len(cluster_df)

    print(f"包含的并购事件数: {event_count} 起")
    print(f"公众平均情感得分: {mean_sentiment:.4f} " + ("(负面情绪较重 🔴)" if mean_sentiment < 0 else "(相对正面 🟢)"))

    # 列出该流派下的代表性公司
    sample_tickers = cluster_df['ticker'].unique()[:5].tolist()  # 最多展示5家
    print(f"代表性企业: {sample_tickers} ...")

print("\n✅ 建模完成！")