const { PythonShell } = require('python-shell');
const { TfidfVectorizer, MultinomialNB } = require('scikit-learn'); // Assuming you're using scikit-learn for training

// Load the trained model from Python
PythonShell.run('load_model.py', null, (err, result) => {
    if (err) throw err;
    const [classifier, vectorizer] = result;
    
    // Now you can use 'classifier' and 'vectorizer' for prediction
    // Example usage:
    const review = "This is a test review";
    const numericalReview = vectorizer.transform([review]);
    const prediction = classifier.predict(numericalReview);
    console.log("Prediction:", prediction[0]);
});
