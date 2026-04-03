This document summarizes the underlying data logic, algorithmic engineering, and core data insights of our project. Our project aims to critically analyze the impact of fintech and corporate discourse on public trust from a humanities perspective, using traditional AI tools such as machine learning, clustering, and text/sentiment analysis. Please refer to the following core content when drafting the Technical Execution and Humanistic Analysis sections of the final report.

1. Data Sourcing
To construct a contrastively rich corpus, we built two aligned datasets via Python scripts:
Official Corporate Voice (Official Text): We called the SEC EDGAR API to bulk-download raw HTML files of Form 8-K filings (material event reports / M&A announcements) submitted by major NASDAQ companies between 2020–2024, and extracted plain text from them.
Public Perception Voice (Public Text & Sentiment): We leveraged an existing database of financial news headlines and social media comments.
Time-Window Alignment: To ensure relevance, we strictly confined the public sentiment collection window to T−1 through T+7 days relative to each official announcement, and computed the average public sentiment score within that window.

2. Data Cleaning & Feature Engineering
During preprocessing, we encountered the classic financial NLP challenge of "Boilerplate Domination."
Cleaning Strategy: SEC filings are saturated with lengthy legal disclaimers (e.g., "Forward-Looking Statements") and meaningless numbers and dates. Direct feature extraction at this stage would obscure the genuine rhetorical content.
Technical Implementation: Using pandas and regular expressions (Regex), we removed all numerals and punctuation, and built a deeply customized financial-domain stop-word list, eliminating hundreds of legal boilerplate terms such as securities, exchange, commission, pursuant, and hereinafter — ensuring that only the purest "corporate narrative" entered the model.

3. Modeling Methodology
Following text cleaning, we used a scikit-learn pipeline to build a traditional AI model combining feature extraction with unsupervised learning:
Text Vectorization (TF-IDF): We applied TfidfVectorizer to extract core feature terms from each official filing, setting ngram_range=(1, 2) to capture key phrases such as "consenting holder."
K-Means Topic Clustering: This is the algorithmic core. We applied unsupervised K-Means clustering (n_clusters=3) to the TF-IDF matrix, allowing the model to autonomously read and categorize the PR rhetoric of these major corporations, ultimately grouping filings into three archetypal "narrative genres."

4. Results & Humanistic Insights
We produced two key visualizations — a box plot and a high-dimensional bubble chart — revealing the profound tension between corporate discourse and public trust. The report should foreground the following three AI-derived narrative clusters and their social-psychological significance:
Cluster 0: Vision & Growth

Key terms: emerging, growth, communications
Analysis: This is the most prevalent narrative style (e.g., Google, Microsoft), yet it yields the lowest public sentiment scores — with strongly negative outliers visible in the bubble chart. This demonstrates that "PR rhetoric has lost its effect." In the public eye, officially proclaimed "emerging growth" is routinely associated with industry monopolization or the workforce reductions that follow. Lengthy vision statements paradoxically deepen social distrust.

Cluster 1: Procedural & Compliance

Key terms: plan, adopted, pre-arranged
Analysis: This narrative style abandons grand vision in favor of dry compliance declarations (e.g., explicitly disclosing executives' pre-arranged stock trading plans). Counterintuitively, it achieves the highest positive sentiment scores — validating the logic of "transparency as trust." The public does not object to capital operations; what it objects to is opacity and backroom dealings.

Cluster 2: Debt & Financing

Key terms: notes, indenture, escrow
Analysis: These documents are dense with hard financial terminology, focusing on the mechanics of fund transfers and debt covenants. They receive moderately positive social evaluations, suggesting that rational financial disclosure — by reducing information asymmetry — is more effective at stabilizing market sentiment than hollow emotional reassurance.