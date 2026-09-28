import joblib

model_data = joblib.load("embedding_classifier.joblib")

classifier = model_data["classifier"]
category_names = model_data["category_names"]
#print(classifier)

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#message = "My parcel still hasn't arrived"



def predict_category(message):
    
    x = embedding_model.encode([message])
    prediction = classifier.predict(x)
    return category_names[prediction[0]]

#print(predict_category(message))