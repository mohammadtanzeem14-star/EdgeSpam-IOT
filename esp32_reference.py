from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.murmurhash import murmurhash3_32

message = "Congratulations! You won a free prize"

vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    lowercase=True
)

analyzer = vectorizer.build_analyzer()
tokens = analyzer(message)

print("Message:")
print(message)

print("\nTokens:")
print(tokens)

print("\nToken -> Feature:")

for token in tokens:
    h = murmurhash3_32(token, seed=0, positive=True)
    feature = h % 512
    print(token, "->", feature)

print("\nThis is the reference the ESP32 must match.")