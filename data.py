messages = [
    "I forgot my password",
    "I cannot login",
    "My account is locked",
    "My card was charged twice",
    "Money was deducted",
    "My payment failed",
    "My package has not arrived",
    "Where is my order",
    "Delivery is very late"
]

labels = [
    "Account","Account","Account",
    "Payment","Payment","Payment",
    "Delivery","Delivery","Delivery"
]
# # vocab is replaced below by .fit(messages) function - To view it we use .get_features_names_out() function
# vocab = set()
# for sentence in messages:
#     words_seq = sentence.lower().split()
#     for word in words_seq:
#         # #no need of below block as set doesn't take duplicates
#         # if word not in vocab:
#         #     vocab.add(word)
#         vocab.add(word)
# vocab = sorted(vocab) #so that words become order and their positions become defined for future iterations.

def word_pres(sentence, word):
    word_ls = sentence.lower().split()
    # #shortcut method below the block
    # for x in word_ls:
    #     if x == word:
    #         return True
    # return False
    return word in word_ls
    
sentence = "I forgot my password"

# #.transform(messages) function will convert every message into a numerical vector using that vocabulary
# vector = []

# for word in vocab:
#     if word_pres(sentence,word):
#         vector.append(1)
#     else:
#         vector.append(0)
# #print(vector)
# #print(len(vector))

from sklearn.feature_extraction.text import CountVectorizer
#It is an object which learns a vocab from sentence and convert sentences into numerical vectors
vectorizer = CountVectorizer(binary=True)

#.fit_transformer is basically fit+transformer
X = vectorizer.fit_transform(messages)

# print(vectorizer.get_feature_names_out())
# print(X.shape)
#print(X)
#print(X.toarray())

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, labels)

new_message = ["Shipment never showed up"]
new_X = vectorizer.transform(new_message)


# print(new_X.toarray())

# print(model.predict_proba(new_X))
# print(model.predict(new_X))
# # print(model.classes_)
# # print(model.intercept_)
# # print(model.coef_.shape)

