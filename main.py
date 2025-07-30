import langchain_helper as lch
import streamlit as st
import textwrap



st.title("YouTube Transcript ")

url = st.text_input("Enter YouTube Video URL:")
query = st.text_input("Enter your question:")


if st.button("Generate Transcript"):
    if url and query:
        try:
            db = lch.Create_vector_db_from_youtube(url)
            response,doc = lch.get_response_from_vector_db(query, db, k=2)
            st.markdown("**Response:**")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid YouTube video URL and your question.")        

