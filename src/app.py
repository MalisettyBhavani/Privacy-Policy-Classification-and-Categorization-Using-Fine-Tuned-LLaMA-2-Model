import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import nltk
nltk.download('punkt')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    model = AutoModelForSequenceClassification.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    return tokenizer, model



def init_page() -> None:
    st.set_page_config(page_title="Chat with Your Fine-Tuned Model")
    st.header("Chat with Your Fine-Tuned Model")
    st.sidebar.title("Options")


def classify_text(tokenizer, model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    # Debugging information
    print(f"Sentence: {input_text}")
    print(f"Logits: {logits}")
    print(f"Predicted Class ID: {predicted_class_id}")
    
    return predicted_class_id

def get_response(tokenizer, model, input_text):
    sentences = nltk.sent_tokenize(input_text)
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    
    response_list = []
    
    for sentence in sentences:
        category_id = classify_text(tokenizer, model, sentence)
        category_name = categories[category_id]
        response_list.append(f"Sentence: {sentence}\nCategory: {category_name}")
    
    response = "\n\n".join(response_list)
    return response

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []

def main() -> None:
    init_page()
    tokenizer, model = load_model()
    init_messages()

    user_input = st.text_area("You:", "")
    if user_input:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(("user", user_input, time.time()))  # Append the current time as a unique identifier
        with st.spinner("Bot is typing ..."):
            response = get_response(tokenizer, model, user_input)
            st.session_state.messages.append(("assistant", response, time.time()))  # Append the current time as a unique identifier

    if "messages" in st.session_state:
        for author, text, timestamp in st.session_state.messages:  # Use the timestamp as part of the key
            key = f"{author}_{timestamp}"  # Create a unique key combining author and timestamp
            if author == "assistant":
                st.text_area("Bot:", value=text, height=200, max_chars=None, key=key)
            else:
                st.text_area("You:", value=text, height=200, max_chars=None, key=key)

if __name__ == "__main__":
    main()



