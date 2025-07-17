import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Giao diện Streamlit
st.set_page_config(page_title="Language Detection App", page_icon="🌐", layout="centered")

# --- Load mô hình đã huấn luyện (đã bao gồm cả TfidfVectorizer + SVC)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model('models/svm_model.pkl')

# --- Hàm tạo biểu đồ Top 5
def create_top5_chart(classes, proba):
    top5_idx = proba.argsort()[-5:][::-1]
    top5_languages = [classes[i] for i in top5_idx]
    top5_probabilities = [proba[i] for i in top5_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top5_languages, [p * 100 for p in top5_probabilities], color='orange')
    ax.set_ylim(0, 100)
    ax.set_title('Top 5 Ngôn ngữ Dự đoán')
    ax.set_ylabel('Xác suất (%)')

    for bar, prob in zip(bars, top5_probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{prob*100:.2f}%', ha='center', va='bottom')

    return fig



with st.sidebar:
    st.header("⚙️ Cài đặt")
    confidence_threshold = st.slider(
        "Ngưỡng độ tin cậy tối thiểu (%)", 0.0, 100.0, 80.0, 1.0,
        help="Chỉ hiển thị kết quả nếu độ tin cậy cao hơn ngưỡng này."
    )
    st.markdown("---")
    st.subheader("ℹ️ Thông tin")
    st.write("Ứng dụng sử dụng mô hình SVM để phát hiện ngôn ngữ dựa trên văn bản đầu vào.")
    st.write("Có thể hỗ trợ tối đa lên đến 20 ngôn ngữ.")
    st.markdown("---")
    st.subheader("📚 Hướng dẫn")
    st.write("- Nhập văn bản vào ô bên dưới.")
    st.write("- Nhấn **Detect Language** để nhận diện.")
    st.write("- Kiểm tra kết quả và top 5 dự đoán.")

st.title("🌐 Language Detection App")
st.markdown("Nhập văn bản vào ô bên dưới để phát hiện ngôn ngữ một cách nhanh chóng và chính xác.")

# --- Nhập văn bản
text_input = st.text_area("📝 Nhập văn bản của bạn:", height=200, placeholder="Nhập hoặc dán văn bản để phát hiện ngôn ngữ...")

# --- Dự đoán khi nhấn nút
if st.button("🔍 Detect Language"):
    if not text_input.strip():
        st.warning("Vui lòng nhập văn bản!", icon="⚠️")
    else:
        with st.spinner("Đang phân tích ngôn ngữ..."):
            proba = model.predict_proba([text_input])[0]
            predicted_lang = model.classes_[proba.argmax()]
            confidence = proba.max() * 100

        if confidence >= confidence_threshold:
            st.success(f"Ngôn ngữ phát hiện: **{predicted_lang}**", icon="✅")
            st.caption(f"Mã '{predicted_lang}' theo chuẩn ISO 639-1")
            st.info(f"Độ tin cậy: **{confidence:.2f}%**", icon="📊")

            # Top 5 + biểu đồ
            st.markdown("### 🔝 Top 5 Ngôn ngữ Dự đoán")
            for lang, p in sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- {lang}: {p:.2%}")

            fig = create_top5_chart(model.classes_, proba)
            st.pyplot(fig)
        else:
            st.error(f"Độ tin cậy ({confidence:.2f}%) thấp hơn ngưỡng ({confidence_threshold}%). Hãy thử văn bản khác!", icon="🚫")

# --- Footer
st.markdown('<div style="text-align: center; margin-top: 2rem; color: gray;">Được phát triển bởi chí dũng | 2025</div>', unsafe_allow_html=True)
