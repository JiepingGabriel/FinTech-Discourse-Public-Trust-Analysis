#final_visualizations.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import numpy as np

print("🚀 初始化数据与模型...")
# 1. 加载与清洗数据
df = pd.read_csv('data/ma_corpus.csv')
df_clean = df.dropna(subset=['public_sentiment_mean', 'public_text']).copy()
df_clean = df_clean[df_clean['public_text_count'] > 0]
df_clean.reset_index(drop=True, inplace=True)

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

# 2. 重新运行 TF-IDF 和 K-Means 以获取标签
tfidf_vectorizer = TfidfVectorizer(stop_words=final_stop_words, max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['clean_official_text'])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)
df_clean['cluster'] = kmeans.labels_

# 为 Cluster 映射我们赋予的“人文学科标签”
cluster_names = {
    0: "Cluster 0: Vision & Growth (愿景与增长)",
    1: "Cluster 1: Procedural & Compliance (程序与合规)",
    2: "Cluster 2: Debt & Financing (资本与债务)"
}
df_clean['cluster_name'] = df_clean['cluster'].map(cluster_names)

# 设置全局绘图风格
sns.set_theme(style="whitegrid", context="talk")

print("📊 正在生成 图1：三大叙事流派的情感分布箱线图...")
# ==========================================
# 图表 1：不同叙事流派的公众情感分布 (Boxplot)
# ==========================================
plt.figure(figsize=(12, 7))
sns.boxplot(x='cluster_name', y='public_sentiment_mean', data=df_clean, palette="Set2")
sns.swarmplot(x='cluster_name', y='public_sentiment_mean', data=df_clean, color=".25", alpha=0.8, size=8)
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Neutral Sentiment')
plt.title('Public Sentiment Distribution Across M&A Rhetoric Schools', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Corporate Rhetoric School (AI Clustered)', fontsize=14)
plt.ylabel('Public Sentiment Score', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('cluster_sentiment_boxplot.png', dpi=300)
plt.close()

print("📊 正在生成 图2：公众参与度、情感与叙事流派的 3D 气泡图...")
# ==========================================
# 图表 2：叙事流派散点气泡图 (Engagement vs Sentiment colored by Cluster)
# ==========================================
plt.figure(figsize=(14, 8))
# 缩放文本长度作为气泡大小
df_clean['bubble_size'] = (df_clean['official_text_len'] / df_clean['official_text_len'].max()) * 1000

sns.scatterplot(
    data=df_clean,
    x='public_text_count',
    y='public_sentiment_mean',
    hue='cluster_name',
    size='bubble_size',
    sizes=(100, 1500),
    palette="Set2",
    alpha=0.7,
    edgecolor='black'
)
plt.axhline(0, color='gray', linestyle='--')
plt.title('M&A Events: Public Engagement vs. Sentiment\n(Color = Rhetoric School, Size = Official Text Length)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Public Mentions (Engagement Level)', fontsize=14)
plt.ylabel('Average Sentiment Score', fontsize=14)
# 优化图例位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('engagement_sentiment_bubble.png', dpi=300)
plt.close()

print("✅ 可视化全部完成！两张高清图表已保存在当前目录。")