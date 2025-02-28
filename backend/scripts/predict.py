import pickle

# Load the trained model and vectorizer
with open("transaction_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def classify_transaction(receiver):
    X_test = vectorizer.transform([receiver])
    category = model.predict(X_test)[0]
    return category

# Test with sample inputs
sample_merchants = ["Amazon", "Swiggy", "BPCL", "Uber", "Apollo Pharmacy"]
for merchant in sample_merchants:
    print(f"Merchant: {merchant} → Category: {classify_transaction(merchant)}")
