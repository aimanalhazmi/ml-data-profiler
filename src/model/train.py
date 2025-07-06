from sklearn import svm
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, model_type: str):
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    elif model_type == 'svm':
        model = svm.SVC(kernel='linear', gamma='auto').fit(X_train, y_train)
    else:
        raise ValueError('Model not supported')
    return model