import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import os
nltk.download('punkt')

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    model = AutoModelForSequenceClassification.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    return tokenizer, model

def init_page() -> None:
    st.set_page_config(page_title="Chat with Your Fine-Tuned Model")
    st.header("Chat with Your Fine-Tuned Model")
    st.sidebar.title("Options")

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from image
def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# Classify text into categories
def classify_text(tokenizer, model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

def get_response(tokenizer, model, input_text):
    sentences = sent_tokenize(input_text)
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    
    classified_sentences = {category: [] for category in categories}
    for sentence in sentences:
        category_id = classify_text(tokenizer, model, sentence)
        category_name = categories[category_id]
        classified_sentences[category_name].append(sentence)
    
    return classified_sentences

# Paraphrasing functionality
@st.cache_resource
def load_paraphraser():
    return pipeline("text2text-generation", model="t5-base")

def paraphrase_text(paraphraser, text, audience):
    if audience == "80-year-old":
        prompt = f"Paraphrase for an 80-year-old: {text}"
    elif audience == "10-year-old":
        prompt = f"Paraphrase for a 10-year-old: {text}"
    result = paraphraser(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]

def main():
    init_page()
    tokenizer, model = load_model()
    paraphraser = load_paraphraser()

    # File upload
    uploaded_file = st.file_uploader("Upload a file (PDF, image, or text)", type=["pdf", "png", "jpg", "jpeg", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            input_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            input_text = extract_text_from_image(uploaded_file)
        elif uploaded_file.type == "text/plain":
            input_text = uploaded_file.read().decode("utf-8")
        st.text_area("Extracted Text:", value=input_text, height=300)

        # Classify text
        classified_sentences = get_response(tokenizer, model, input_text)
        
        # Display categories as buttons
        st.sidebar.subheader("Categories")
        for category, sentences in classified_sentences.items():
            if st.sidebar.button(category):
                st.write(f"**{category}**")
                for sentence in sentences:
                    st.write(f"- {sentence}")

        # Paraphrasing options
        st.sidebar.subheader("Paraphrase Options")
        if st.sidebar.button("Paraphrase for 80-year-old"):
            paraphrased = paraphrase_text(paraphraser, input_text, "80-year-old")
            st.write("**Paraphrased for 80-year-old:**")
            st.write(paraphrased)
        
        if st.sidebar.button("Paraphrase for 10-year-old"):
            paraphrased = paraphrase_text(paraphraser, input_text, "10-year-old")
            st.write("**Paraphrased for 10-year-old:**")
            st.write(paraphrased)

if __name__ == "__main__":
    main()
