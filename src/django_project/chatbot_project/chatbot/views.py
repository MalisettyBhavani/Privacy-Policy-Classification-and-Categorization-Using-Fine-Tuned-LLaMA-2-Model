from django.shortcuts import render
from django import forms
from . import forms
from django.http import HttpResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import nltk
import torch.nn.functional as F  # For softmax

nltk.download('punkt')

def ChatbotInputForm_View(request):
    if request.method=="POST":
        text = request.POST.get('text')
        print(text) 
        input_data = text
        output_data = llm_response(input_data)
        return render(request,"chatbot_form.html",{"input":input_data,"output":output_data})
    return render(request, "chatbot_form.html")

def display_message():
    return HttpResponse("Hello World")

def llm_response(input_text):
    prompt = """You are a highly specialized privacy policy classifier.
            As an expert in categorizing privacy policy texts your task is to accurately classify the provided text into the most suitable category.
            Additionally, ensure you consider whether multiple categories apply when necessary. Be thorough, and ensure that the output is clear and concise.
            Please use your expertise to categorize the provided privacy policy text into the following categories: First Party Collection/Use, Third Party Sharing/Collection, Other, User Choice/Control, Do Not Track, International and Specific Audiences, Data Security, Policy Change, Data Retention, User Access, Edit and Deletion."""
    
    prompt = prompt + input_text

    tokenizer = AutoTokenizer.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    model = AutoModelForSequenceClassification.from_pretrained('/home/bhavani/GA Work/Fine_Tuning_and_ChatBot_Creation/src/FineTuning_11/my-llama2-chat-hf-llm')
    sentences = nltk.sent_tokenize(input_text)
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    
    response_list = []
    
    for sentence in sentences:
        full_prompt = prompt + "\n\n" + sentence
        category_id = classify_text(tokenizer, model, full_prompt)
        category_name = categories[category_id]
        response_list.append(f"Sentence: {sentence}\nCategory: {category_name}")
    response = "\n\n".join(response_list)
    return response

def classify_text(tokenizer, model, input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).detach().numpy()[0]

    predicted_class_id = logits.argmax().item()
    
    # Map the predicted class ID to the corresponding category
    categories = [
        'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
        'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
        'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
    ]
    predicted_category = categories[predicted_class_id]
    
    print(f"Input Text: {input_text}")
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted Class ID: {predicted_class_id}")
    print(f"Predicted Category: {predicted_category}")
    
    
    return predicted_class_id
