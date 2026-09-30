def evaluate_classification(y_true, y_pred, y_score=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try: result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError: pass
        try: result["pr_auc"] = float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(y_true, y_score))
        except ValueError: pass
    return result


def evaluate_regression(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "mse": float(mse), "rmse": float(mse ** .5), "r2": float(r2_score(y_true, y_pred))}


def evaluate_clustering(X, labels):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    return {"silhouette": float(silhouette_score(X, labels)), "davies_bouldin": float(davies_bouldin_score(X, labels)), "calinski_harabasz": float(calinski_harabasz_score(X, labels))}
