from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "free money now",
    "win lottery today",
    "hello how are you",
    "let us meet tomorrow"
]

labels = [1,1,0,0]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

model = MultinomialNB()

model.fit(X, labels)

new_email = ["free lottery win"]

new_X = vectorizer.transform(new_email)

prediction = model.predict(new_X)

print(prediction)