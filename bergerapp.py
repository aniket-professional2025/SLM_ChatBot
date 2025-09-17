# Importing Required Packages
import streamlit as st
from bergermain import get_answer
import time

# The streamlit App Interface
st.set_page_config(page_title = "Fine-Tuned SLM Chatbot", layout = "wide")
st.title("Fine-Tuned SLM Chatbot")

# Setting the user prompt
user_input = st.text_area("Enter your message here:", placeholder = 'Ask me anything')

# Place the Refresh button after the submit button's logic
if st.button("Refresh", type = "primary"):
    # This will clear the input area and rerun the script
    st.rerun()

# Button setting and defining the action
if st.button("Submit Question") and user_input:
    # Create an empty placeholder for the status messages
    status_placeholder = st.empty()

    with st.spinner("Generating Response..."):
        # Display the status messages one by one
        status_placeholder.info("Loading the User Question.....")
        time.sleep(3)

        status_placeholder.info("Tokenizing the User Question....")
        time.sleep(3)

        status_placeholder.info("Sending the Tokenized Question into the SLM...")
        time.sleep(3)

        status_placeholder.info("Fetching the Tokenized answer that matches the question....")
        time.sleep(3)

        status_placeholder.info("Decoding the Tokenized Answer....")
        time.sleep(3)

        # Get the actual answer from your model
        response = get_answer(user_input)

        # Once the process is complete, update the placeholder with the final message
        status_placeholder.info("Answer Received Successfully")
        time.sleep(3)

    # Display the final generated response
    st.header("Answer")
    st.markdown(response)