from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from data import messages, labels

# Load pretrained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#Convert our 9 training messages into embeddings
X = embedding_model.encode(messages)

# #print(embeddings.shape)



# similarity_1 = cosine_similarity(
#     [embeddings[0]],
#     [embeddings[1]]
# )

# similarity_2 = cosine_similarity(
#     [embeddings[0]],
#     [embeddings[2]]
# )

# print(similarity_1)
# print(similarity_2)

#print (X.shape)

#Train classifier
classifier = LogisticRegression()
classifier.fit(X, labels)

new_message = ["Shipment never showed up"]

new_X = embedding_model.encode(new_message)

prediction = classifier.predict(new_X)
probabilities = classifier.predict_proba(new_X)

print(classifier.classes_)
print(probabilities)
print(prediction)