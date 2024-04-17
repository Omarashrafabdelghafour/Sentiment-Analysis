import pickle

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    classifier, vectorizer = pickle.load(f)

# Return the model objects
print(classifier, vectorizer)

from sklearn.externals import joblib

# Assuming 'classifier' and 'vectorizer' are your trained model objects
joblib.dump(classifier, 'classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
