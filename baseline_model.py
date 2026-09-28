from inspect_dataset import split_dataset
X_train = split_dataset["train"]['instruction']
y_train = split_dataset["train"][:]['category']

X_test = split_dataset["test"]['instruction']
y_test = split_dataset["test"]['category']

# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))

# print(X_train[0])
# print(y_train[0])

# category_feature = split_dataset['train'].features["category"]
# print(category_feature.int2str(y_train[0]))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)

X_train_vectorized = vectorizer.fit_transform(X_train)

# print(X_train_vectorized.shape)
# print(X_test_vectorized.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized,y_train)


# things that should happen ONLY when this file is run directly

if __name__ == "__main__":
    X_test_vectorized = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vectorized)

    # print(len(y_pred))
    # print(y_pred[:10])
    # print(y_test[:10])

    # #Accuracy manual code
    # correct = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] == y_test[i]:
    #         correct+=1

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)    

    #print('accuracy', correct/len(y_pred))


    #Error Analysis
    category_feature = split_dataset['train'].features["category"]

    # for i in range(len(y_pred)):
    #     if y_pred[i]!=y_test[i]:
    #         print(f"Message: {X_test[i]}\n")
    #         print(f"Actual: {category_feature.int2str(y_test[i])}\n")
    #         print(f"Pred: {category_feature.int2str(int(y_pred[i]))}\n")

    from sklearn.metrics import classification_report, confusion_matrix

    target_names = []
    for i in range(11):
        name = category_feature.int2str(i)
        target_names.append(name)

    #print(target_names)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    print(cm.shape)
