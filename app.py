import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud

st.set_page_config(page_title="Twitter Sentiment Dashboard", page_icon="🐦", layout="wide")


# Load stopwords
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))


# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Text preprocessing
def preprocess(text, stop_words):

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()

    text = [word for word in text if word not in stop_words]

    return " ".join(text)


# Sentiment prediction
def predict_sentiment(text, model, vectorizer, stop_words):

    clean_text = preprocess(text, stop_words)

    vector = vectorizer.transform([clean_text])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


# Word cloud generator
def generate_wordcloud(text):

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)


# Pie chart
def sentiment_pie_chart(pos, neg):

    labels = ["Positive", "Negative"]
    values = [pos, neg]

    fig, ax = plt.subplots()

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)


# Main dashboard
def main():

    st.title("🐦 Twitter Sentiment Analysis Dashboard")
    st.write("Analyze tweet sentiment using **Machine Learning + NLP**")

    stop_words = load_stopwords()
    model, vectorizer = load_model()

    st.divider()

    st.subheader("Paste Tweets (One per line)")

    user_input = st.text_area(
        "Example:\nI love AI\nThis product is terrible\nAmazing technology"
    )

    if st.button("Analyze Sentiment"):

        tweets = user_input.split("\n")

        positive = 0
        negative = 0

        results = []

        for tweet in tweets:

            if tweet.strip() == "":
                continue

            sentiment = predict_sentiment(
                tweet,
                model,
                vectorizer,
                stop_words
            )

            results.append((tweet, sentiment))

            if sentiment == "Positive":
                positive += 1
            else:
                negative += 1

        st.divider()

        # RESULTS PANEL
        st.subheader("Sentiment Results")

        for tweet, sentiment in results:

            if sentiment == "Positive":
                st.success(f"{tweet} → 😊 Positive")

            else:
                st.error(f"{tweet} → 😡 Negative")

        st.divider()

        # ANALYTICS PANEL
        st.subheader("Interactive Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Sentiment Distribution")
            sentiment_pie_chart(positive, negative)

        with col2:
            st.write("### Tweet Word Cloud")

            all_text = " ".join([tweet for tweet, _ in results])

            generate_wordcloud(all_text)

        st.divider()

        st.write("### Analytics Summary")

        st.metric("Positive Tweets", positive)
        st.metric("Negative Tweets", negative)


if __name__ == "__main__":
    main()