from baseline_model import vectorizer, model

from embedding_model import embedding_model, embedding_classifier

from inspect_dataset import category_feature

custom_messages = [
    "Amazon charged me but the checkout page says the payment failed",
    "Been waiting two weeks and the parcel still isn't here",
    "Where can I get a copy of what I was billed for?",
    "I moved houses yesterday, where do I update where you send my stuff?",
    "Take my card off this account",
    "I want my money back",
    "Your product is terrible",
    "I can't get into my profile"
]

X1 = vectorizer.transform(custom_messages)
y1 = model.predict(X1)

X2 = embedding_model.encode(custom_messages)
y2 = embedding_classifier.predict(X2)

print(f"bow {y1}\n")
print(f"embedding {y2}\n")

for i in range(len(custom_messages)):
    print (f"Message: {custom_messages[i]}")
    print(f"BoW: {category_feature.int2str(int(y1[i]))}")
    print(f"MiniLM: {category_feature.int2str(int(y2[i]))}\n")
      