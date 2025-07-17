import pandas as pd
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Tải dữ liệu punkt cho tokenization
#nltk.download('punkt', quiet=True)

# Ánh xạ mã ngôn ngữ sang tên đầy đủ
LANGUAGE_MAP = {
    'pt': 'Portuguese', 'bg': 'Bulgarian', 'zh': 'Chinese', 'th': 'Thai', 'ru': 'Russian',
    'ja': 'Japanese', 'en': 'English', 'de': 'German', 'es': 'Spanish', 'fr': 'French',
    'vi': 'Vietnamese', 'hi': 'Hindi', 'ar': 'Arabic', 'it': 'Italian', 'ko': 'Korean',
    'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'ro': 'Romanian'
}

# Chuẩn hoá UNICODE
def normalize_unicode(text):
    return unicodedata.normalize("NFKC", str(text))

# Tách từ văn bản 
#def tokenize_text(text):
    tokens = word_tokenize(text)
    return " ".join(tokens)

# Xử lí văn bản, bỏ dấu, bỏ viết hoa, bỏ số
def preprocess_text(text):
    text = normalize_unicode(text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    #text = tokenize_text(text)
    return text

# Áp dụng tiền xử lý cho cột văn bản trong DataFrame
def preprocess_dataframe(df, text_column='text'):
    df['Processed_Text'] = df[text_column].apply(preprocess_text)
    return df

# Tạo và huấn luyện TF-IDF vectorizer
def create_tfidf_vectorizer(texts, ngram_range=(1, 3), max_features=5000):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    return vectorizer, X_tfidf

# Chuyển mã ngôn ngữ
def map_language_code_to_name(code):
    return LANGUAGE_MAP.get(code, code)