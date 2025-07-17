import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from utils.preprocessing import preprocess_dataframe

def train_model(data_path, model_path='models/svm_model.pkl'):
    # Đọc dữ liệu
    df = pd.read_csv(data_path)
    print(f"Số lượng dòng: {df.shape[0]}, Số lượng cột: {df.shape[1]}")

    # Xử lý dữ liệu thiếu và trùng lặp
    if df.isnull().sum().sum() > 0:
        print("=> Phát hiện dữ liệu thiếu, tiến hành xử lý...")
        df = df.dropna()

    if df.duplicated().sum() > 0:
        print(f"Xóa {df.duplicated().sum()} dòng trùng lặp...")
        df = df.drop_duplicates()

    # Tiền xử lý văn bản
    df = preprocess_dataframe(df, text_column='text')

    # Tạo pipeline gồm vectorizer và SVC
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SVC(kernel='linear', probability=True, random_state=42)
    )

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed_Text'], df['labels'], test_size=0.2, random_state=42
    )

    # RandomizedSearchCV
    random_params = {'svc__C': np.logspace(-3, 3, 100)}
    random_search = RandomizedSearchCV(pipeline, random_params, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # GridSearchCV tiếp tục tối ưu từ kết quả tốt nhất
    best_C = random_search.best_params_['svc__C']
    grid_params = {'svc__C': np.linspace(best_C * 0.5, best_C * 1.5, 5)}
    grid_search = GridSearchCV(pipeline, grid_params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Đánh giá mô hình
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy trên tập test: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # ✅ Lưu pipeline (đã bao gồm vectorizer đã fit + SVC)
    joblib.dump(best_model, model_path)
    print(f"✅ Mô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    data_path = "data/train.csv"
    train_model(data_path)
