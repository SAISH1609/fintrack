from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle  # Used to save the trained model

# Sample data: Mapping of merchant names to categories
categories = {
    "Amazon": "Shopping",
    "Flipkart": "Shopping",
    "Swiggy": "Food",
    "Zomato": "Food",
    "BPCL": "Petrol",
    "Google Pay": "General",
    "Myntra": "Shopping",
    "Uber": "Transport",
    "Ola": "Transport",
    "Apollo Pharmacy": "Medical",
    "BigBasket": "Grocery",
    "Reliance Fresh": "Grocery"
}

# Step 1: Prepare training data
train_data = list(categories.keys())  # Merchant names
train_labels = list(categories.values())  # Corresponding categories

# Step 2: Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)

# Step 3: Train a Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train, train_labels)

# Step 4: Save the trained model and vectorizer for later use
with open("transaction_classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved successfully!")
