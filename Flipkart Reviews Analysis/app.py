from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pickle

# Download NLTK's VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load the pickled model
model_path = (r'C:\Users\DELL\Downloads\New folder (2)\FlipkartReview.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        reviewer = request.form.get("reviewer_name")
        product = request.form.get("product_name")
        review = request.form.get("review")

        if not review:
            message = 'Enter review.'
            return render_template('home.html', message=message)
        else:
            print("Review:", review)  # Print the review data
            try:
                # Get sentiment score for the review
                sentiment_score = analyzer.polarity_scores(review)
                # Determine sentiment from the scores
                prediction = 'positive' if sentiment_score['compound'] >= 0 else 'negative'
                return render_template('home.html', prediction=prediction, sentiment_score=sentiment_score)
            except Exception as e:
                print("Prediction error:", e)  # Print any prediction errors
                return render_template('home.html', message="Error occurred during prediction. Please try again.")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
