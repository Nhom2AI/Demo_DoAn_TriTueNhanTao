import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# =====================
# 1. LOAD DATA
# =====================
try:
    df = pd.read_csv("data.csv")
except:
    print("❌ Không tìm thấy file data.csv")
    exit()

# Kiểm tra dữ liệu
if 'text' not in df.columns or 'label' not in df.columns:
    print("❌ File CSV phải có cột: text,label")
    exit()

print("✅ Đã load dữ liệu")
print("\nPhân bố nhãn:")
print(df['label'].value_counts())

# =====================
# 2. CLEAN TEXT
# =====================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# =====================
# 3. SPLIT DATA
# =====================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
except:
    print("⚠️ Không đủ dữ liệu để stratify → dùng chia thường")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

# =====================
# 4. TF-IDF
# =====================
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =====================
# 5. MODEL 1: NAIVE BAYES
# =====================
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

print("\n===== Naive Bayes =====")
print(classification_report(y_test, y_pred_nb, zero_division=0))

# =====================
# 6. MODEL 2: LOGISTIC REGRESSION
# =====================
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

print("\n===== Logistic Regression =====")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# =====================
# 7. MODEL 3: SVM
# =====================
svm = SVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

print("\n===== SVM =====")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# =====================
# 8. DEMO THỰC TẾ
# =====================
print("\n===== DEMO DỰ ĐOÁN =====")

samples = [
    "Có học sinh đánh nhau ngoài sân",
    "Wifi trường rất yếu",
    "Em không hiểu bài hôm nay",
    "Có cháy trong phòng học"
]

for s in samples:
    s_clean = clean_text(s)
    vec = vectorizer.transform([s_clean])
    result = lr.predict(vec)[0]
    print(f"{s}  -->  {result}")