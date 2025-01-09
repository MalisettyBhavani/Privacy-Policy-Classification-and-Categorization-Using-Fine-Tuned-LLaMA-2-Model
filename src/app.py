import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import nltk
nltk.download('punkt')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    model = AutoModelForSequenceClassification.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm', num_labels = 10)
    return tokenizer, model

def init_page() -> None:
    st.set_page_config(page_title="Chat with Your Fine-Tuned Model")
    st.header("Chat with Your Fine-Tuned Model")
    st.sidebar.title("Options")

def classify_text(tokenizer, model, input_text):
    list_of_categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    
    # Include the structured prompt
    prompt = (
        "As a privacy policy classification AI, your task is to classify sections of privacy policies into one or more categories based on the information provided. "
        "You are specialized in identifying the right category or multiple categories from complex privacy policy texts. "
        "Ensure that your classifications are accurate, comprehensive, and consistent with the provided privacy policy excerpt. "
        "Based on the provided privacy policy text, please classify it into the following categories: "
        "First Party Collection/Use, Third Party Sharing/Collection, Other, User Choice/Control, Do Not Track, International and Specific Audiences, "
        "Data Security, Policy Change, Data Retention, User Access, Edit and Deletion. "
        "Make sure to account for situations where multiple categories might apply.\n\n"
        f"Privacy Policy Text: {input_text}\n\nClassification:"
    )
    
    # Tokenize with the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()  # Convert logits to probabilities
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_category = list_of_categories[predicted_class_id]
    
    # Debugging information
    print(f"Prompt:\n{prompt}")
    print(f"Sentence: {input_text}")
    print(f"Logits: {logits}")
    print(f"Logits shape: {logits.shape}") 
    print(f"Probabilities: {probabilities}")
    print(f"Predicted Class ID and category: {predicted_class_id} {predicted_category}")
    
    return predicted_class_id, probabilities


def get_response(tokenizer, model, input_text):
    sentences = nltk.sent_tokenize(input_text)
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    
    response_list = []
    
    for sentence in sentences:
        predicted_class_id, probabilities = classify_text(tokenizer, model, sentence)
        category_name = categories[predicted_class_id]
        
        # Add detailed category probabilities to the response
        category_probabilities = "\n".join(
            [f"{categories[i]}: {prob:.4f}" for i, prob in enumerate(probabilities)]
        )
        
        response_list.append(
            f"Sentence: {sentence}\nPredicted Category: {category_name}\nCategory Probabilities:\n{category_probabilities}"
        )
    
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



