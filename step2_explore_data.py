import urllib.request
import zipfile
import os
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_FILE = "sms_spam.zip"
DATA_FILE = "SMSSpamCollection"

# Download dataset if not present
if not os.path.exists(DATA_FILE):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")

    print("Dataset extracted!")

# Load dataset
df = pd.read_csv(
    DATA_FILE,
    sep="\t",
    names=["label", "text"]
)

# Display information
print("\n--- Dataset Summary ---")
print("Total messages:", len(df))

print("\nHam and Spam counts:")
print(df["label"].value_counts())

print("\nSample Ham Message:")
print(df[df["label"] == "ham"]["text"].iloc[0])

print("\nSample Spam Message:")
print(df[df["label"] == "spam"]["text"].iloc[0])