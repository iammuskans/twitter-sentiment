🐦 Twitter Sentiment Analysis 

An interactive Machine Learning dashboard built with Python and Streamlit that analyzes the sentiment of tweets or text input using Natural Language Processing (NLP).

The application classifies text into Positive 😊 or Negative 😡 sentiment and provides visual analytics such as sentiment distribution charts and word clouds.

🚀 Features

Sentiment prediction using a trained Machine Learning model
Text preprocessing using NLP techniques
Interactive Streamlit dashboard
Sentiment Pie Chart visualization
Tweet Word Cloud for keyword insights
Real-time analytics panel
Clean and user-friendly interface


🧠 Machine Learning Workflow

Text Input
↓
Text Preprocessing (Cleaning, Lowercasing, Stopword Removal)
↓
TF-IDF Vectorization
↓
Machine Learning Model Prediction
↓
Sentiment Output + Visual Analytics

🛠️ Tech Stack

Python
Streamlit
Scikit-learn
NLTK
WordCloud
Matplotlib

📊 Dashboard Components

The dashboard includes:

Sentiment Classifier – Predicts sentiment of each tweet/text
Sentiment Pie Chart – Shows distribution of positive vs negative tweets
Word Cloud Visualization – Displays frequently used words
Interactive Analytics Panel – Displays sentiment statistics

📂 Project Structure
twitter-sentiment
│
├── app.py                # Streamlit dashboard
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

🌟 Future Improvements

Support for Neutral sentiment
Integration with live Twitter/X API
Advanced NLP models such as BERT
Real-time social media analytics
