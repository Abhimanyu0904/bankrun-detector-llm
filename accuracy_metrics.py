def generate_accuracy_metrics(confusion_matrix):
    # Unpack confusion matrix values
    TP, FP, FN, TN = confusion_matrix

    # Calculate accuracy, precision, recall, f1-score
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print("Confusion Matrix")
    print("         Predicted:    ")
    print("              Pos    Neg")
    print(f"Actual: Pos {TP:5} {FN:5}")
    print(f"        Neg {FP:5} {TN:5}")

    # Print each statistic
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}\n")