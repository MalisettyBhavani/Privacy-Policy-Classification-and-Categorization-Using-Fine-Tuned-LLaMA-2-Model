import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfFolder
import re  
import pandas as pd
from util.generating_cm import compute_confusion_matrix_technique1,compute_grouped_metrics
import time
import os
class PrivacyPolicyClassifier:
    def __init__(self, model_path=None, hf_token=None):
        self.list_of_categories = [
            'First Party Collection/Use', 'Third Party Sharing/Collection', 'Other',
            'User Choice/Control', 'Do Not Track', 'International and Specific Audiences',
            'Data Security', 'Policy Change', 'Data Retention', 'User Access, Edit and Deletion'
        ]
        self.category_results_technique1 = {category: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for category in self.list_of_categories}
        self.actual_labels_list = []
        self.predicted_labels_list = []

        # Model loading based on provided parameters
        if model_path:
            # Load fine-tuned or quantized model from local directory
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        elif hf_token:
            # Save the token and load the LLaMA 2 model from Hugging Face
            HfFolder.save_token(hf_token)
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            self.model = pipeline("text-generation", model=model_name, tokenizer=self.tokenizer,
                                  torch_dtype=torch.float16, device_map="auto")
        else:
            raise ValueError("Either 'model_path' or 'hf_token' must be provided to load a model.")

    def predict_category(self, prompt_template, text):
        prompt = prompt_template + " " + text + "\n\nCategory:"
        num_prompt_tokens = len(self.tokenizer.encode(prompt))  # Count tokens in the prompt

        if hasattr(self.model, "generate"):  # For local models
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')

            # Start timing
            start_time = time.time()
            with torch.no_grad():
                output = self.model.generate(inputs, max_new_tokens=20)  # Generate response
            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time  # Time in seconds
            predicted_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            total_tokens = len(self.tokenizer.encode(predicted_text))  # Count total tokens in the output
            generated_tokens = self.tokenizer.tokenize(predicted_text[len(prompt):].strip())
        
            new_tokens = len(generated_tokens)  # Update new_tokens to visible token count
            generated_text = predicted_text[len(prompt):].strip()  # Extract only the generated portion
            predicted_categories = generated_text.split('Category:')[-1].strip().split(', ')
            print(f"Generated new tokens: {generated_text}")
            print(self.tokenizer.tokenize(generated_text))
            tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0

        else:  # For Hugging Face pipeline model
            start_time = time.time()
            predicted_text = self.model(prompt, do_sample=True, top_k=10, num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id, max_length=2060)
            end_time = time.time()

            elapsed_time = end_time - start_time
            total_tokens = len(self.tokenizer.encode(predicted_text[0]['generated_text']))  # Count total tokens in the output

            new_tokens = total_tokens - num_prompt_tokens  # Count only new tokens generated
            generated_text = predicted_text[0]['generated_text'][len(prompt):].strip()  # Extract only the generated portion
            category_text = None
            if "Explanation:" in generated_text:
                category_text = generated_text.split("Explanation:")[0].strip()
            elif "Reason:" in generated_text:
                category_text = generated_text.split("Reason:")[0].strip()
            elif "Reasoning:" in generated_text:
                category_text = generated_text.split("Reasoning:")[0].strip()
            else:
                category_text = generated_text.strip()

            # Clean up and extract the category from the category text
            if "Category:" in category_text:
                category_text = category_text.split("Category:")[-1].strip()

            predicted_categories = self.extract_category(category_text)
            tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
            print(f"Generated new tokens:{generated_text}")
            print(self.tokenizer.tokenize(generated_text))



        # Print the new tokens and their generation rate
        print(f"Elapsed Time: {elapsed_time}")
        print(f"New tokens generated: {new_tokens}")
        print(f"Tokens generated per second: {tokens_per_second:.2f} tokens/s")
        print(f"Predicted categories:{predicted_categories}")
        return predicted_categories, tokens_per_second, new_tokens

    def extract_category(self, text):
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.list_of_categories)) + r')\b', re.IGNORECASE)
        matches = pattern.findall(text)
        return matches if matches else ["Unknown Category"]
    
    def predict_and_evaluate(self, prompt_template, test_data_path, num_test_data=10, model_type=None, prompt_number=None):
        with open(test_data_path, 'r') as file:
            test_data = json.load(file)

        num_test_data = min(num_test_data, len(test_data))
        test_data = test_data[:num_test_data]

        tokens_per_second_list = []  # List to store tokens per second for each prediction
        new_tokens_list = []  # List to store the number of new tokens generated for each prediction

        for item in test_data:
            privacy_policy_text = item["text"]
            actual_labels = item["label"]
            
            # Call predict_category to get predictions, elapsed time, and new tokens generated
            predicted_categories, elapsed_time, new_tokens = self.predict_category(prompt_template, privacy_policy_text)
            
            tokens_per_second = new_tokens / elapsed_time if elapsed_time > 0 else 0
            tokens_per_second_list.append(tokens_per_second)
            new_tokens_list.append(new_tokens)
            
            print(f"Privacy Policy Text: {privacy_policy_text}")
            print(f"Actual Category: {actual_labels}")
            print(f"Predicted Categories: {predicted_categories}")
            print()

            # Convert actual and predicted categories to binary labels for evaluation
            actual_binary_labels = [int(cat in actual_labels) for cat in self.list_of_categories]
            predicted_binary_labels = [int(cat in predicted_categories) for cat in self.list_of_categories]

            self.actual_labels_list.append(actual_binary_labels)
            self.predicted_labels_list.append(predicted_binary_labels)
        
        # Compute metrics for Technique 1
        technique1_metrics = compute_confusion_matrix_technique1(self)
        
        # Compute grouped metrics
        grouped_metrics = compute_grouped_metrics(self)

        # Calculate and print the average new tokens and tokens per second
        average_new_tokens = sum(new_tokens_list) / len(new_tokens_list)
        average_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
        print(f"New tokens list: {new_tokens_list}")
        print(f"Tokens per second list: {tokens_per_second_list}")
        print(f"Average New Tokens Generated: {average_new_tokens:.2f}")
        print(f"Average Tokens Processed per Second: {average_tokens_per_second:.2f} tokens/s")

    # Save results to Excel
        if model_type and prompt_number is not None:
            print("\n=== Saving Results to Excel ===")
            self.save_results_to_excel(
                model_type,
                prompt_number,
                technique1_metrics,
                grouped_metrics,
                average_tokens_per_second,
                average_new_tokens
            )
            print("\n=== Results Saved Successfully ===")

        return self

    def group_data(self):
        group_1_actual = []
        group_1_predicted = []
        group_2_actual = []
        group_2_predicted = []

        for actual, predicted in zip(self.actual_labels_list, self.predicted_labels_list):
            actual_count = sum(actual)
            predicted_count = sum(predicted)

            if actual_count == 1 and predicted_count == 1:
                group_1_actual.append(actual)
                group_1_predicted.append(predicted)
            else:
                group_2_actual.append(actual)
                group_2_predicted.append(predicted)

        return group_1_actual, group_1_predicted, group_2_actual, group_2_predicted

    def save_results_to_excel(self, model_type, prompt_number, technique1_metrics, grouped_metrics, average_tokens_per_second, average_new_tokens):
        # Create base directory for results
        base_directory = "results"
        os.makedirs(base_directory, exist_ok=True)  # Ensure the "results" directory exists

        # Create model-specific subdirectory
        directory_name = os.path.join(base_directory, model_type.lower().replace(" ", "_"))
        os.makedirs(directory_name, exist_ok=True)  # Ensure the model-specific directory exists

        # Define file name
        file_name = os.path.join(directory_name, f"prompt_{prompt_number}_results.xlsx")

        rows = []

        rows.append([
            f"Prompt {prompt_number} - Technique 1",
            technique1_metrics["accuracy"],
            technique1_metrics["precision"],
            technique1_metrics["recall"],
            technique1_metrics["f1"],
            average_tokens_per_second,
            average_new_tokens
        ])

        rows.append([
            f"Prompt {prompt_number} - G1",
            grouped_metrics["G1"]["accuracy"],
            grouped_metrics["G1"]["precision"],
            grouped_metrics["G1"]["recall"],
            grouped_metrics["G1"]["f1"],
            average_tokens_per_second,
            average_new_tokens
        ])

        rows.append([
            f"Prompt {prompt_number} - G2",
            grouped_metrics["G2"]["accuracy"],
            grouped_metrics["G2"]["precision"],
            grouped_metrics["G2"]["recall"],
            grouped_metrics["G2"]["f1"],
            average_tokens_per_second,
            average_new_tokens
        ])

        rows.append([
            f"Prompt {prompt_number} - G1+G2",
            grouped_metrics["G1+G2"]["accuracy"],
            grouped_metrics["G1+G2"]["precision"],
            grouped_metrics["G1+G2"]["recall"],
            grouped_metrics["G1+G2"]["f1"],
            average_tokens_per_second,
            average_new_tokens
        ])

        columns = [
            "Category", "Accuracy", "Precision", "Recall", "F1-Score", 
            "Average Tokens/s", "Average New Tokens"
        ]

        new_data = pd.DataFrame(rows, columns=columns)

        # Check if the file exists and handle it
            # Save the new data, overwriting any existing file
        print(f"Saving results to '{file_name}'. Overwriting if it already exists.")
        new_data.to_excel(file_name, index=False)
        print(f"Results saved to {file_name} for Prompt {prompt_number}")