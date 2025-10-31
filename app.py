# app.py
import streamlit as st
import torch
from transformers import BertTokenizerFast, BertModel
from model_class import SentimentClassifier

# --------------------------
# Load Model and Tokenizer
# --------------------------
@st.cache_resource  # cache for efficiency
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Load BERT encoder
    bert_encoder = BertModel.from_pretrained("bert-base-uncased")

    # Initialize custom classifier
    model = SentimentClassifier(bert_encoder, num_labels=3)

    # Load saved weights
    model.load_state_dict(torch.load("model.pt", map_location=device))

    model.to(device)
    model.eval()
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --------------------------
# Prediction Function
# --------------------------
def predict_sentiment(texts):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    label_map_inv = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [label_map_inv[p] for p in preds]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")

st.title("📝 Sentiment Analysis")
st.markdown("""
Welcome! Enter customer feedback or text below, and this app will predict its **sentiment**:
- Positive 😄
- Neutral 😐
- Negative 😢
""")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing..."):
            predictions = predict_sentiment([user_input])
            sentiment = predictions[0]

        # Display result with friendly UI
        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment} 😄")
        elif sentiment == "Negative":
            st.error(f"Predicted Sentiment: {sentiment} 😢")
        else:
            st.info(f"Predicted Sentiment: {sentiment} 😐")

# --------------------------
# Optional: Example Texts
# --------------------------
st.markdown("---")
st.subheader("Try Example Texts")
example_texts = [
    "I loved the product, it was amazing!",
    "The service was terrible and slow.",
    "It was okay, nothing special."
]

for text in example_texts:
    if st.button(f"Test: {text}"):
        with st.spinner("Analyzing..."):
            sentiment = predict_sentiment([text])[0]
        st.write(f"**Text:** {text}")
        st.write(f"**Predicted Sentiment:** {sentiment}")
