class Evaluator:
    def __init__(self, target_class="Positive"):
        """
        Initializes the evaluator. By default, we evaluate metrics focusing 
        on how well the model predicts the 'Positive' class.
        """
        self.target_class = target_class

    def compute_confusion_matrix(self, true_labels: list, pred_labels: list):
        """
        Calculates the raw counts for the confusion matrix.
        """
        tp = 0  # True Positives: Predicted Positive, Actually Positive
        fp = 0  # False Positives: Predicted Positive, Actually Negative
        tn = 0  # True Negatives: Predicted Negative, Actually Negative
        fn = 0  # False Negatives: Predicted Negative, Actually Positive
        
        for true, pred in zip(true_labels, pred_labels):
            # Normalize strings to avoid case-sensitivity bugs
            true_clean = str(true).capitalize()
            pred_clean = str(pred).capitalize()
            
            if true_clean == self.target_class and pred_clean == self.target_class:
                tp += 1
            elif true_clean != self.target_class and pred_clean == self.target_class:
                fp += 1
            elif true_clean == self.target_class and pred_clean != self.target_class:
                fn += 1
            elif true_clean != self.target_class and pred_clean != self.target_class:
                tn += 1
                
        return tp, fp, tn, fn

    def calculate_metrics(self, true_labels: list, pred_labels: list) -> dict:
        """
        Computes Accuracy, Precision, Recall, and the F1-Score from scratch 
        using safe division to prevent ZeroDivisionError.
        """
        tp, fp, tn, fn = self.compute_confusion_matrix(true_labels, pred_labels)
        
        # 1. Accuracy: Overall correctness
        total_predictions = tp + fp + tn + fn
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0
        
        # 2. Precision: Out of all predicted Positives, how many were actually Positive?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 3. Recall: Out of all actual Positives, how many did the model successfully find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 4. F1-Score: The harmonic mean of Precision and Recall
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        return {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1_score, 4),
            "Confusion_Matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        }
        
    def display_report(self, model_name: str, true_labels: list, pred_labels: list):
        """
        Generates a clean terminal output for the model's performance, 
        mirroring the standard scikit-learn classification report format.
        """
        metrics = self.calculate_metrics(true_labels, pred_labels)
        
        print(f"========================================")
        print(f" Performance Report: {model_name}")
        print(f"========================================")
        print(f" Accuracy:  {metrics['Accuracy']:.4f}")
        print(f" Precision: {metrics['Precision']:.4f}")
        print(f" Recall:    {metrics['Recall']:.4f}")
        print(f" F1-Score:  {metrics['F1_Score']:.4f}")
        print(f"----------------------------------------")
        print(f" Confusion Matrix: ")
        print(f" TP: {metrics['Confusion_Matrix']['TP']:<5} | FP: {metrics['Confusion_Matrix']['FP']}")
        print(f" FN: {metrics['Confusion_Matrix']['FN']:<5} | TN: {metrics['Confusion_Matrix']['TN']}")
        print(f"========================================\n")
        
        return metrics