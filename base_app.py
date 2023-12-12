"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Visualisation dependencies
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/tfidf_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Infinite  ∞  innovation")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction","EDA", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with SVM Classifier ML Model")
		# Creating a text box for user input
		
		tweet_text = st.text_area("Enter Text","Type Here", key="user_input")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/svm_classifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	# Building the EDA page
	if selection == "EDA":
		st.info("Exploratory Data Analysis")


		# # Display the distribution of sentimentsst.subheader("Distribution of Sentiments")
		st.subheader("distribution of sentiments")
		sentiment_counts = raw['sentiment'].value_counts()
		st.bar_chart(sentiment_counts)

		# Display a word cloud of the most common words in messages
		from wordcloud import WordCloud
		st.subheader("Word Cloud of Messages")
		wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(raw['message']))
		st.image(wordcloud.to_array())

		# Display a histogram of message lengths
		st.subheader("Sistribution of Message Lengths")
		raw['message_length'] = raw['message'].apply(len)
		plt.hist(raw['message_length'], bins=30, edgecolor='k')
		st.pyplot()


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
