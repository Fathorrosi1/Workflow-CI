# MLProject/modelling.py
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path ke dataset')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training
    with mlflow.start_run(run_name='ci_training'):
        # Log params
        mlflow.log_param('n_estimators', args.n_estimators)
        mlflow.log_param('max_depth', args.max_depth)
        mlflow.log_param('model_type', 'RandomForestClassifier')
        
        # Model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrik
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Log metrics
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        
        # Artefak 1: Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        mlflow.log_artifact('confusion_matrix.png')
        
        # Artefak 2: Feature Importance
        plt.figure(figsize=(10, 8))
        importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=True).tail(15)
        importances.plot(kind='barh', color='steelblue')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        mlflow.log_artifact('feature_importance.png')
        
        # Artefak 3: Classification Report
        report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report.txt')
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            serialization_format='pickle',
            skops_trusted_types=['sklearn.tree._tree.Tree']
        )
        
        print("Training selesai!")

if __name__ == '__main__':
    main()
