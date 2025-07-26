import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("BERT Sentiment Analysis")
review = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        inputs = tokenizer(review, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        score = probs[0][pred_class].item()
        star_rating = pred_class + 1
        if star_rating <= 2:
            label = 'Negative'
        elif star_rating == 3:
            label = 'Neutral'
        else:
            label = 'Positive'
        st.success(f"Prediction: **{label}** ({star_rating} stars, Confidence: {score:.2f})")
