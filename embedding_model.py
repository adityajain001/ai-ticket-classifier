from inspect_dataset import split_dataset
from sentence_transformers import SentenceTransformer
import joblib

X_train = split_dataset["train"]["instruction"]
y_train = split_dataset["train"]["category"]

X_test = split_dataset["test"]["instruction"]
y_test = split_dataset["test"]["category"]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#print(len(X_train),len(X_test))

X_train_encode= embedding_model.encode(X_train)

#print(X_train_encode.shape, X_test_encode.shape)

from sklearn.linear_model import LogisticRegression

embedding_classifier = LogisticRegression()
embedding_classifier.fit(X_train_encode, y_train)


category_feature = split_dataset['train'].features["category"]
category_names = []
for i in range(11):
    category_names.append(category_feature.int2str(i))

model_data = {

    "classifier": embedding_classifier,

    "category_names": category_names

}

joblib.dump(model_data, "embedding_classifier.joblib")
#print(category_names)

if __name__ == "__main__":
    X_test_encode = embedding_model.encode(X_test)
    y_pred = embedding_classifier.predict(X_test_encode)

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)   

    category_feature = split_dataset['train'].features["category"]

    # for i in range(len(y_pred)):
    #     if y_pred[i]!=y_test[i]:
    #         print(f"Message: {X_test[i]}\n")
    #         print(f"Actual: {category_feature.int2str(y_test[i])}\n")
    #         print(f"Pred: {category_feature.int2str(int(y_pred[i]))}\n")

