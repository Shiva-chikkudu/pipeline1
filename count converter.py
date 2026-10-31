from sklearn.feature_extraction.text import CountVectorizer
from sklearn. naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn. preprocessing import LabelEncoder

emails = [
    "congratulations! you're selected as a employee",
    "urgent: your account will be closed",
    "Tresspassers will be Prosecuted",
    "Notice: Your accout has been Freezed for a certain period  of time",
    "Primary shopping alert: your order has been shipped"
]
 
accounts = [
    "PalNati company",
    "Sunnala Sumit",
    "Government of state Prohibition",
    "CENTRAL BANK OF  INDIA",
    "State Bank of India"
] 

labels = [1, 3, 5, 4, 1]
Vectorizer = CountVectorizer()
X = Vectorizer.fit_transform(emails)
Y = Vectorizer.fit_transform(accounts)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=52)
model = MultinomialNB()
Vectorizer = CountVectorizer()
X = Vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, random_state=52
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

new_email = ["Your account has been credited with $19000"]
prediction = model.predict(Vectorizer.transform(new_email))

print("Prediction:", prediction[0])
print("Accuracy:", accuracy)
new_email = ["Your account has been credited with $19000"]
new_X = Vectorizer.transform(new_email)
new_account = ["The account of under the Name with Manual Recognition from the User"]
new_Y = Vectorizer.transform(new_account)
prediction = model.predict(new_X)
label_encoded = LabelEncoder()
label_encoded.fit(labels)
accuracy = accuracy_score(label_encoded.transform(y_test), label_encoded.transform(predictions))
print("Prediction:", prediction[labels.index(1)])
print("Accuracy:", accuracy)
