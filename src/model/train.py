from sklearn import svm
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, model_type: str):
    """
    Train a classification model based on specified model type.
    
    This function supports training either a Logistic Regression model or 
    a Linear Support Vector Machine (SVM) classifier.
    
    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training target vector (n_samples,)
        model_type: Type of model to train. Supported values: "logistic" or "svm"
        
    Returns:
        Trained classifier model
        
    Raises:
        ValueError: If an unsupported model type is provided
        
    Example:
        model = train_model(X_train, y_train, "logistic")
    """
    if model_type == "logistic":
        model = LogisticRegression(solver="saga", max_iter=10000).fit(X_train, y_train)
        # st.info("Logistic Regression model trained")
    elif model_type == "svm":
        model = svm.SVC(kernel="linear", gamma="auto").fit(X_train, y_train)
        # st.info("SVM model trained")
    else:
        raise ValueError("Model not supported")
    return model
