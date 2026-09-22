from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    lowercase=True
)

message = "Congratulations! You won a free prize"

features = vectorizer.transform([message])

print("Message:")
print(message)

print("\nNon-zero features:")

for index, value in zip(features.indices, features.data):
    print("Feature:", index, "Value:", value)

print("\nTotal features:", features.shape[1])
print("Non-zero count:", features.nnz)