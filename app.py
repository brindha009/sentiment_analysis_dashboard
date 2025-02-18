import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

@st.cache_data
def load_data():
    return pd.read_csv('Instruments_Reviews.csv')

def classify_sentiment(overall):
    if overall > 3:
        return 'Positive'
    elif overall == 3:
        return 'Neutral'
    else:
        return 'Negative'

data = load_data()

# Adding a dynamic classification on-the-fly
data['sentiment'] = data['overall'].apply(classify_sentiment)

# Streamlit UI components
st.sidebar.title("Sentiment Analysis Dashboard")
sample_size = st.sidebar.slider("Select number of samples", min_value=100, max_value=len(data), step=100)
sentiment_filter = st.sidebar.multiselect("Filter by sentiment", options=['Positive', 'Neutral', 'Negative'], default=['Positive', 'Neutral', 'Negative'])

if st.sidebar.button("Analyze Sentiments"):
    st.write("Dataset Preview")
    st.dataframe(data.head(sample_size))

    st.write("Sentiment Distribution")
    sentiment_counts = data[data['sentiment'].isin(sentiment_filter)]['sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    st.write("Word Cloud")
    selected_data = data[data['sentiment'].isin(sentiment_filter)]
    text = ' '.join(review for review in selected_data['reviews'])
    wordcloud = WordCloud(max_words=100, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# User input for sentiment analysis
user_input = st.text_area("Enter text for sentiment analysis", "Type your text here...")
if user_input and st.button("Analyze"):
    # Perform sentiment analysis using TextBlob
    analysis = TextBlob(user_input)
    sentiment_polarity = analysis.sentiment.polarity
    sentiment = 'Positive' if sentiment_polarity > 0 else ('Neutral' if sentiment_polarity == 0 else 'Negative')
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Polarity Score: {sentiment_polarity:.2f}")
