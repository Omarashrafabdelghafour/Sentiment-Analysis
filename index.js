const express = require('express');
const app = express();
const port = 3000; // Choose a port number

const nltk = require('nltk');
const { SentimentIntensityAnalyzer } = nltk.sentiment;

app.use(express.json());

app.post('/predict', (req, res) => {
  try {
    const review = req.body.review; // Extract the review text from the request body

    // Perform any necessary preprocessing on the review text

    // Create an instance of the SentimentIntensityAnalyzer
    const sentimentAnalyzer = new SentimentIntensityAnalyzer();

    // Use the analyzer to get sentiment scores
    const sentimentScore = sentimentAnalyzer.polarity_scores(review);

    // Return the sentiment scores or desired output
    res.json({ sentiment: sentimentScore });
  } catch (error) {
    // Handle the error
    console.error('An error occurred:', error.message);
    res.status(500).json({ error: 'An error occurred' });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});