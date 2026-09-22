from sklearn.feature_extraction.text import HashingVectorizer

message = "Congratulations! You won a free prize"

vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    lowercase=True
)

features = vectorizer.transform([message])

print("Python reference features:")

for index, value in zip(features.indices, features.data):
    print(index, value)