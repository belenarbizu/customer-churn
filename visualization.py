import joblib
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import tree
import graphviz


def open_model(file_path):
    model = joblib.load(file_path)
    return model


def open_info(metrics_path, params_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    with open(params_path, 'r') as f:
        params = json.load(f)
    return metrics, params


def open_predictions(file_path):
    pred = pd.read_csv(file_path)
    return pred


def roc_visualization(pred):
    # fpr: false positive rate, tpr: true positive rate
    fpr, tpr, _ = roc_curve(pred['y_true'], pred['y_proba'])
    roc_auc = auc(fpr, tpr)

    if not os.path.exists('images'):
        os.makedirs('images')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('images\\roc_curve.png')
    plt.close()


def confusion_matrix_visualization(pred):
    cm = confusion_matrix(pred['y_true'], pred['y_pred'])
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig('images\\confusion_matrix.png')
    plt.close()


def feature_importances(model):
    feature_importances = pd.Series(model.named_steps['classifier'].feature_importances_, index=model.named_steps['classifier'].feature_names_in_)
    feature_importances.sort_values(ascending=False, inplace=True)
    
    plt.figure(figsize=(10, 6))
    feature_importances[:10].plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('images\\feature_importances.png')
    plt.close()


def tree_visualization(model):
    tree_data = model.named_steps['classifier'].estimators_[0]
    data = tree.export_graphviz(tree_data, out_file=None, feature_names=model.named_steps['classifier'].feature_names_in_, filled=True, proportion=True, max_depth=3)
    graph = graphviz.Source(data)
    graph.render('images/decision_tree', format='png')


def main():
    model = open_model('models\\model.pkl')
    metrics, params = open_info('models\\metrics.json', 'models\\best_params.json')
    pred = open_predictions('models\\predictions.csv')
    roc_visualization(pred)
    confusion_matrix_visualization(pred)
    feature_importances(model)
    tree_visualization(model)


if __name__ == "__main__":
    main()