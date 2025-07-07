from sklearn import svm
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, model_type: str):
    if model_type == "logistic":
        model = LogisticRegression(solver="saga", max_iter=10000).fit(X_train, y_train)
        # st.info("Logistic Regression model trained")
    elif model_type == "svm":
        model = svm.SVC(kernel="linear", gamma="auto").fit(X_train, y_train)
        # st.info("SVM model trained")
    else:
        raise ValueError("Model not supported")
    return model
