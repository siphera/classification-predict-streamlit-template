"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Infinite Innovation

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
import pickle 
import os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidf_vectorizer.pkl","rb")
tweet_cv = pickle.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Set overall theme
	st.set_page_config(
		page_title="Climate Change Sentiment Analysis",
		page_icon="🌍",
	)

	# Customize colors and layout
	st.markdown(
		"""
		<style>
		.main {
			background-color: #3e5243;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Infinite  ∞  innovation")

	# Add a logo to the sidebar
	logo_path = "resources/imgs/logo.png"
	st.sidebar.image(logo_path, use_column_width=True)

	st.subheader("Climate change tweet classification 🌍")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction","EDA", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("App Overview")
		st.markdown(
            """
            Welcome to the **Infinite ∞ Innovation** Climate Change Sentiment Analysis app! 🌍

            **Purpose:**
            This application aims to analyze sentiments expressed in tweets related to climate change.
            By leveraging machine learning models, we categorize tweets into sentiment classes, providing insights
            into public opinions and discussions surrounding climate issues.

            **Models:**
            Our app employs state-of-the-art machine learning models, including Support Vector Machines (SVM),
            Convolutional Neural Networks (CNN), and variations thereof. These models have been trained on a diverse
            dataset of climate change tweets to accurately classify sentiments.

            **Data:**
            The models were trained on a curated dataset sourced from diverse social media platforms.
            The dataset contains a variety of messages, enabling our models to capture the nuanced expressions of
            opinions regarding climate change.

            **How to Use:**
            1. Navigate to the **Prediction** tab.
            2. Enter a tweet in the text area.
            3. Choose a model from the dropdown menu.
            4. Click the "Classify" button to see the sentiment prediction.

            **Model Performance:**
            We continuously strive to enhance model performance. Metrics such as accuracy, precision, and recall
            are regularly assessed. While the models are powerful, it's important to consider their strengths and
            limitations.

            Thank you for using the Infinite ∞ Innovation Climate Change Sentiment Analysis app! 🌎✨
            """
        )
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with Different Machine Learning Models")
		# Creating a text box for user input
		
		tweet_text = st.text_area("Enter Text","Type Here", key="user_input")

		# Model selection dropdown
		model_options = ["SVM", "CNN", "CNN2"]
		selected_model = st.selectbox("Choose Model", model_options)

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice

			# Load the selected model
			if selected_model == "SVM":
				predictor = pickle.load(open(os.path.join("resources/svm_classifier2.pkl"), "rb"))
				prediction = predictor.predict(vect_text)
			elif selected_model == "CNN":
				predictor = pickle.load(open(os.path.join("resources/lsvc_classifier.pkl"), "rb"))
				prediction = predictor.predict(vect_text)
			elif selected_model == "CNN2":
				predictor = pickle.load(open(os.path.join("resources/svm_classifier.pkl"), "rb"))
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
		# from wordcloud import WordCloud
		# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(raw['message']))
		# st.image(wordcloud.to_array())

		# Load and display the pre-generated word cloud image
		st.subheader("Word Cloud of Messages")
		wordcloud_image_path = "resources/imgs/wordcloud.jpg"
		st.image(wordcloud_image_path, use_column_width=True)

		# Load and display the pre-generated Frequent words image
		st.subheader("Frequent words from each sentiment")
		frequnt_image_path = "resources/imgs/frequent_words.png"
		st.image(frequnt_image_path, use_column_width=True)

		# Load and display the pre-generated popula hashtags image
		st.subheader("Popular #hashtags from each sentiment")
		hashtag_image_path = "resources/imgs/Hashtag_analysis.png"
		st.image(hashtag_image_path, use_column_width=True)



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
