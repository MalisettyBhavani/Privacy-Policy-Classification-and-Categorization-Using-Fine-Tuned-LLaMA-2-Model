import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Technique 1: Generating confusion matrix without dividing into groups (multi-label classification)
def compute_confusion_matrix_technique1(classifier):
    print("Computing confusion matrix using Technique 1 (Multi-Label Classification)...")

    y_true = np.array(classifier.actual_labels_list)
    y_pred = np.array(classifier.predicted_labels_list)

    # Iterate over all categories to compute TP, FP, FN, TN for each category
    for i, category in enumerate(classifier.list_of_categories):
        for j in range(len(y_true)):
            actual = y_true[j][i]
            predicted = y_pred[j][i]

            if actual == 1 and predicted == 1:
                # True Positive: both actual and predicted are category x
                classifier.category_results_technique1[category]["TP"] += 1
            elif actual == 1 and predicted == 0:
                # False Negative: actual is category x, but predicted is not
                classifier.category_results_technique1[category]["FN"] += 1
            elif actual == 0 and predicted == 1:
                # False Positive: actual is not category x, but predicted is
                classifier.category_results_technique1[category]["FP"] += 1
            elif actual == 0 and predicted == 0:
                # True Negative: both actual and predicted are not category x
                classifier.category_results_technique1[category]["TN"] += 1

    # After computing, print the confusion matrix for each category
    for category, results in classifier.category_results_technique1.items():
        plot_confusion_matrix_for_category(category, results["TP"], results["FP"], results["FN"], results["TN"])

    # Compute overall metrics
    metrics = compute_overall_metrics_technique1(classifier)
    return metrics  # Return metrics as a dictionary

# Function to compute and print overall metrics for Technique 1
def compute_overall_metrics_technique1(classifier):
    overall_tp = sum(res["TP"] for res in classifier.category_results_technique1.values())
    overall_fp = sum(res["FP"] for res in classifier.category_results_technique1.values())
    overall_fn = sum(res["FN"] for res in classifier.category_results_technique1.values())
    overall_tn = sum(res["TN"] for res in classifier.category_results_technique1.values())
    overall_cases = overall_tp + overall_fp + overall_fn + overall_tn

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = (overall_tp + overall_tn) / overall_cases if overall_cases > 0 else 0
    
    print("\nOverall tp:",overall_tp)
    print("Overall tn:",overall_tn)
    print("Overall fp:",overall_fp)
    print("Overall fn:",overall_fn)
    print(f"Overall Precision (Technique 1): {overall_precision}")
    print(f"Overall Recall (Technique 1): {overall_recall}")
    print(f"Overall F1 Score (Technique 1): {overall_f1}")
    print(f"Overall Accuracy (Technique 1): {overall_accuracy}")

    # Micro average: Sum TP / (Sum TP + Sum FP)
    micro_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    micro_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    micro_accuracy = (overall_tp + overall_tn) / (overall_tp + overall_fp + overall_fn + overall_tn)

    # Macro average: Average of precision, recall, and F1 scores over all categories
    macro_precision = np.mean([res["TP"] / (res["TP"] + res["FP"]) if (res["TP"] + res["FP"]) > 0 else 0 for res in classifier.category_results_technique1.values()])
    macro_recall = np.mean([res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0 for res in classifier.category_results_technique1.values()])
    macro_f1_values = []
    for res in classifier.category_results_technique1.values():
        precision = res["TP"] / (res["TP"] + res["FP"]) if (res["TP"] + res["FP"]) > 0 else 0
        recall = res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        macro_f1_values.append(f1)

    macro_f1 = np.mean(macro_f1_values)
    
    # Weighted average: Precision/recall/f1 for each category weighted by the number of actual positives (TP + FN) in that category
    category_support = [res["TP"] + res["FN"] for res in classifier.category_results_technique1.values()]
    total_support = sum(category_support)
    weighted_precision = np.sum([(res["TP"] / (res["TP"] + res["FP"]) if (res["TP"] + res["FP"]) > 0 else 0) * support for res, support in zip(classifier.category_results_technique1.values(), category_support)]) / total_support
    weighted_recall = np.sum([(res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0) * support for res, support in zip(classifier.category_results_technique1.values(), category_support)]) / total_support
    weighted_f1_values = []
    for res, support in zip(classifier.category_results_technique1.values(), category_support):
        precision = res["TP"] / (res["TP"] + res["FP"]) if (res["TP"] + res["FP"]) > 0 else 0
        recall = res["TP"] / (res["TP"] + res["FN"]) if (res["TP"] + res["FN"]) > 0 else 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        weighted_f1_values.append(f1 * support)

    weighted_f1 = np.sum(weighted_f1_values) / total_support if total_support > 0 else 0

    sample_precisions = []
    sample_recalls = []
    sample_f1s = []
    
    for actual, predicted in zip(classifier.actual_labels_list, classifier.predicted_labels_list):
        # Convert actual and predicted to numpy arrays for easier manipulation
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # True Positives (TP): correctly predicted positive labels
        tp = np.sum(np.logical_and(actual == 1, predicted == 1))
        
        # False Positives (FP): predicted positive, but actual is negative
        fp = np.sum(np.logical_and(actual == 0, predicted == 1))
        
        # False Negatives (FN): actual positive, but predicted as negative
        fn = np.sum(np.logical_and(actual == 1, predicted == 0))
        
        # Precision: TP / (TP + FP)
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0  # If there are no positive predictions, precision is 0
        
        # Recall: TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0  # If there are no actual positives, recall is 0
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0  # If both precision and recall are 0, F1 score is also 0
        
        # Append the results for this sample
        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1s.append(f1)

    for res in classifier.category_results_technique1.values():
        print("TP",res["TP"])
        print("TN",res["TN"])
        print("FP",res["FP"])
        print("FN",res["FN"])
        print()
  
    # Calculate average metrics
    sample_precision_avg = np.mean(sample_precisions)
    sample_recall_avg = np.mean(sample_recalls)
    sample_f1_avg = np.mean(sample_f1s)

    


    # Print all results
    print(f"Micro Precision: {micro_precision}")
    print(f"Micro Recall: {micro_recall}")
    print(f"Micro F1 Score: {micro_f1}")
    print(f"Micro Accuracy: {micro_accuracy}")
    
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall: {macro_recall}")
    print(f"Macro F1 Score: {macro_f1}")
    
    print(f"Weighted Precision: {weighted_precision}")
    print(f"Weighted Recall: {weighted_recall}")
    print(f"Weighted F1 Score: {weighted_f1}")
    
    # Print the final sample average metrics
    print(f"Sample Average Precision: {sample_precision_avg}")
    print(f"Sample Average Recall: {sample_recall_avg}")
    print(f"Sample Average F1 Score: {sample_f1_avg}")

    print("\nOverall Metrics (Technique 1):")
    print(f"Accuracy: {overall_accuracy}, Precision: {overall_precision}, Recall: {overall_recall}, F1 Score: {overall_f1}")

    return {"accuracy": overall_accuracy, "precision": overall_precision, "recall": overall_recall, "f1": overall_f1}

# Function to plot confusion matrix for each category in Technique 1
def plot_confusion_matrix_for_category(category, TP, FP, FN, TN):
    print(f"\nPlotting confusion matrix for category: {category}")
    matrix = np.array([[TN, FP], [FN, TP]])

    plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix for {category} (Technique 1)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to compute and plot confusion matrices for Group 1 and Group 2
def compute_confusion_matrix(classifier, group_1_results, group_2_results):
    y_true = np.array(classifier.actual_labels_list)
    y_pred = np.array(classifier.predicted_labels_list)

    overall_cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    # Plot combined confusion matrix (overall)
    plt.figure(figsize=(10, 8))
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classifier.list_of_categories, yticklabels=classifier.list_of_categories)
    plt.xlabel('Predicted Categories')
    plt.ylabel('Actual Categories')
    plt.title('Combined Confusion Matrix (All Groups)')
    plt.show()

    
    # Print metrics for Group 1 and Group 2
    print(f"\nMetrics for Group 1:")
    print(f"Accuracy: {group_1_results['accuracy']}")
    print(f"Precision: {group_1_results['precision']}")
    print(f"Recall: {group_1_results['recall']}")
    print(f"F1 Score: {group_1_results['f1']}")

    print(f"\nMetrics for Group 2:")
    print(f"Accuracy: {group_2_results['accuracy']}")
    print(f"Precision: {group_2_results['precision']}")
    print(f"Recall: {group_2_results['recall']}")
    print(f"F1 Score: {group_2_results['f1']}")

    # Now plot confusion matrices for Group 1
    print("\nConfusion Matrices for Group 1:")
    plot_group_confusion_matrices(group_1_results['category_results'], classifier.list_of_categories, 'Group 1')

    # Now plot confusion matrices for Group 2
    print("\nConfusion Matrices for Group 2:")
    plot_group_confusion_matrices(group_2_results['category_results'], classifier.list_of_categories, 'Group 2')


# Function to handle confusion matrix for Group 1
def compute_metrics_and_confusion_matrix_group_1(classifier, y_true, y_pred):
    category_results = {category: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for category in classifier.list_of_categories}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Iterate and calculate TP, FP, FN, TN for Group 1
    for i in range(len(y_true)):
        actual_label = list(np.where(y_true[i] == 1)[0])[0]
        predicted_label = list(np.where(y_pred[i] == 1)[0])[0]

        print(f"\n--- Group 1 Sample {i+1} ---")
        print(f"Actual Label: {classifier.list_of_categories[actual_label]}")
        print(f"Predicted Label: {classifier.list_of_categories[predicted_label]}")

        # True Positive
        if actual_label == predicted_label:
            category_results[classifier.list_of_categories[actual_label]]["TP"] += 1
            print(f"Incremented TP for {classifier.list_of_categories[actual_label]}")
        else:
            # False Negative for actual, False Positive for predicted
            category_results[classifier.list_of_categories[actual_label]]["FN"] += 1
            category_results[classifier.list_of_categories[predicted_label]]["FP"] += 1
            print(f"Incremented FN for {classifier.list_of_categories[actual_label]}")
            print(f"Incremented FP for {classifier.list_of_categories[predicted_label]}")

        # True Negatives for others
        for idx in range(len(classifier.list_of_categories)):
            if idx != actual_label and idx != predicted_label:
                category_results[classifier.list_of_categories[idx]]["TN"] += 1
                print(f"Incremented TN for {classifier.list_of_categories[idx]}")

    # After processing, print final category results
    print("\nFinal category results for Group 1:")
    for category, results in category_results.items():
        print(f"{category}: TP={results['TP']}, FP={results['FP']}, FN={results['FN']}, TN={results['TN']}")

    # Aggregating TP, FP, FN, and TN values across all categories for Group 1
    aggregated_tp = sum([category_results[cat]["TP"] for cat in classifier.list_of_categories])
    aggregated_fp = sum([category_results[cat]["FP"] for cat in classifier.list_of_categories])
    aggregated_fn = sum([category_results[cat]["FN"] for cat in classifier.list_of_categories])
    aggregated_tn = sum([category_results[cat]["TN"] for cat in classifier.list_of_categories])

    # Print aggregated values for Group 1
    print(f"\nAggregated TP (Group 1): {aggregated_tp}")
    print(f"Aggregated FP (Group 1): {aggregated_fp}")
    print(f"Aggregated FN (Group 1): {aggregated_fn}")
    print(f"Aggregated TN (Group 1): {aggregated_tn}")

    # Calculate accuracy, precision, recall, and F1 score using the aggregated values
    accuracy = (aggregated_tp + aggregated_tn) / (aggregated_tp + aggregated_fp + aggregated_fn + aggregated_tn) if (aggregated_tp + aggregated_fp + aggregated_fn + aggregated_tn) > 0 else 0
    precision = aggregated_tp / (aggregated_tp + aggregated_fp) if (aggregated_tp + aggregated_fp) > 0 else 0
    recall = aggregated_tp / (aggregated_tp + aggregated_fn) if (aggregated_tp + aggregated_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nFinal Metrics for Group 1:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "category_results": category_results}


# Function to handle confusion matrix for Group 2
def compute_metrics_and_confusion_matrix_group_2(classifier, y_true, y_pred):
    category_results = {category: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for category in classifier.list_of_categories}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Iterate and calculate TP, FP, FN, TN for Group 2
    for i in range(len(y_true)):
        actual_labels = set(np.where(y_true[i] == 1)[0])
        predicted_labels = set(np.where(y_pred[i] == 1)[0])
        union_labels = actual_labels.union(predicted_labels)
        intersection_labels = actual_labels.intersection(predicted_labels)

        print(f"\n--- Group 2 Sample {i+1} ---")
        print(f"Actual Labels: {[classifier.list_of_categories[idx] for idx in actual_labels]}")
        print(f"Predicted Labels: {[classifier.list_of_categories[idx] for idx in predicted_labels]}")
        print(f"Union Labels: {[classifier.list_of_categories[idx] for idx in union_labels]}")
        print(f"Intersection Labels: {[classifier.list_of_categories[idx] for idx in intersection_labels]}")

        # If there's an intersection (at least one correct prediction)
        if intersection_labels:
            for idx in union_labels:
                category_results[classifier.list_of_categories[idx]]["TP"] += 1
                print(f"Incremented TP for {classifier.list_of_categories[idx]}")
        else:
            # Increment False Negatives for actual, False Positives for predicted
            for idx in actual_labels:
                category_results[classifier.list_of_categories[idx]]["FN"] += 1
                print(f"Incremented FN for {classifier.list_of_categories[idx]}")
            for idx in predicted_labels:
                category_results[classifier.list_of_categories[idx]]["FP"] += 1
                print(f"Incremented FP for {classifier.list_of_categories[idx]}")

        # True Negatives for others
        for idx in range(len(classifier.list_of_categories)):
            if idx not in union_labels:
                category_results[classifier.list_of_categories[idx]]["TN"] += 1
                print(f"Incremented TN for {classifier.list_of_categories[idx]}")

    # After processing, print final category results
    print("\nFinal category results for Group 2:")
    for category, results in category_results.items():
        print(f"{category}: TP={results['TP']}, FP={results['FP']}, FN={results['FN']}, TN={results['TN']}")

    # Aggregating TP, FP, FN, and TN values across all categories for Group 2
    aggregated_tp = sum([category_results[cat]["TP"] for cat in classifier.list_of_categories])
    aggregated_fp = sum([category_results[cat]["FP"] for cat in classifier.list_of_categories])
    aggregated_fn = sum([category_results[cat]["FN"] for cat in classifier.list_of_categories])
    aggregated_tn = sum([category_results[cat]["TN"] for cat in classifier.list_of_categories])

    # Print aggregated values for Group 2
    print(f"\nAggregated TP (Group 2): {aggregated_tp}")
    print(f"Aggregated FP (Group 2): {aggregated_fp}")
    print(f"Aggregated FN (Group 2): {aggregated_fn}")
    print(f"Aggregated TN (Group 2): {aggregated_tn}")

    # Calculate accuracy, precision, recall, and F1 score using the aggregated values
    accuracy = (aggregated_tp + aggregated_tn) / (aggregated_tp + aggregated_fp + aggregated_fn + aggregated_tn) if (aggregated_tp + aggregated_fp + aggregated_fn + aggregated_tn) > 0 else 0
    precision = aggregated_tp / (aggregated_tp + aggregated_fp) if (aggregated_tp + aggregated_fp) > 0 else 0
    recall = aggregated_tp / (aggregated_tp + aggregated_fn) if (aggregated_tp + aggregated_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nFinal Metrics for Group 2:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "category_results": category_results}

# Function to plot individual confusion matrices
def plot_group_confusion_matrices(category_results, list_of_categories, group_name):
    for category in list_of_categories:
        cm_data = category_results[category]
        cm = np.array([[cm_data["TN"], cm_data["FP"]], [cm_data["FN"], cm_data["TP"]]])

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {category} ({group_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# Function to compute overall accuracy, precision, recall, and f1 score for combined Group 1 and Group 2
def compute_combined_metrics(classifier, group_1_results, group_2_results):
    print("\nComputing combined overall metrics for Group 1 and Group 2...")

    # Summing up TP, FP, FN, and TN from Group 1 and Group 2
    combined_tp = 0
    combined_fp = 0
    combined_fn = 0
    combined_tn = 0
    combined_category_results = {}

    for category in classifier.list_of_categories:
        combined_tp += group_1_results['category_results'][category]["TP"] + group_2_results['category_results'][category]["TP"]
        combined_fp += group_1_results['category_results'][category]["FP"] + group_2_results['category_results'][category]["FP"]
        combined_fn += group_1_results['category_results'][category]["FN"] + group_2_results['category_results'][category]["FN"]
        combined_tn += group_1_results['category_results'][category]["TN"] + group_2_results['category_results'][category]["TN"]
        combined_category_results[category] = {
            "TP": group_1_results['category_results'][category]["TP"] + group_2_results['category_results'][category]["TP"],
            "FP": group_1_results['category_results'][category]["FP"] + group_2_results['category_results'][category]["FP"],
            "FN": group_1_results['category_results'][category]["FN"] + group_2_results['category_results'][category]["FN"],
            "TN": group_1_results['category_results'][category]["TN"] + group_2_results['category_results'][category]["TN"]
        }
    # Calculating overall accuracy, precision, recall, and F1 score
    total_cases = combined_tp + combined_fp + combined_fn + combined_tn

    accuracy = (combined_tp + combined_tn) / total_cases if total_cases > 0 else 0
    precision = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else 0
    recall = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print the combined metrics
    print(f"\nCombined Metrics for Group 1 and Group 2:")
    print(f"Overall Accuracy: {accuracy}")
    print(f"Overall Precision: {precision}")
    print(f"Overall Recall: {recall}")
    print(f"Overall F1 Score: {f1_score}")

    print(f"Combined Metrics -> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1_score}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score}
# Function to compute and plot combined confusion matrices for each category (Group 1 + Group 2)
def plot_combined_confusion_matrices_for_each_category(classifier, group_1_results, group_2_results):
    print("\nPlotting combined confusion matrices for each category (Group 1 + Group 2)...")

    # Iterate over each category
    for category in classifier.list_of_categories:
        # Combine TP, FP, FN, TN from Group 1 and Group 2
        combined_tp = group_1_results['category_results'][category]["TP"] + group_2_results['category_results'][category]["TP"]
        combined_fp = group_1_results['category_results'][category]["FP"] + group_2_results['category_results'][category]["FP"]
        combined_fn = group_1_results['category_results'][category]["FN"] + group_2_results['category_results'][category]["FN"]
        combined_tn = group_1_results['category_results'][category]["TN"] + group_2_results['category_results'][category]["TN"]

        # Create the combined confusion matrix for the category
        combined_cm = np.array([[combined_tn, combined_fp], [combined_fn, combined_tp]])

        # Plot the combined confusion matrix using Seaborn heatmap
        plt.figure(figsize=(4, 4))
        sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Combined Confusion Matrix for {category} (Group 1 + Group 2)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Print out the combined values for the category
        print(f"\n{category} - Combined Confusion Matrix (Group 1 + Group 2):")
        print(f"TP: {combined_tp}, FP: {combined_fp}, FN: {combined_fn}, TN: {combined_tn}")


def compute_grouped_metrics(classifier):
    group_1_actual, group_1_predicted, group_2_actual, group_2_predicted = classifier.group_data()

    # Process Group 1
    print("\nProcessing Group 1")
    group_1_results = compute_metrics_and_confusion_matrix_group_1(classifier, group_1_actual, group_1_predicted)

    # Process Group 2
    print("Processing Group 2")
    group_2_results = compute_metrics_and_confusion_matrix_group_2(classifier, group_2_actual, group_2_predicted)

    # Compute combined confusion matrix
    compute_confusion_matrix(classifier, group_1_results, group_2_results)

    # Compute the combined overall accuracy, precision, recall, and F1 score
    combined_metrics = compute_combined_metrics(classifier, group_1_results, group_2_results)
    
    # Plot combined confusion matrices for each category (Group 1 + Group 2)
    plot_combined_confusion_matrices_for_each_category(classifier, group_1_results, group_2_results)

    return {
        "G1": group_1_results,
        "G2": group_2_results,
        "G1+G2": combined_metrics
    }

