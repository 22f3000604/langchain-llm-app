import langchain_helper as lch
import streamlit as st
import time

st.title("Pet Name Generator")

pet_type = st.text_input("Enter the type of pet (e.g., cat, dog):")
number_of_names = 5

style = st.text_input("Enter the style of names (e.g., elegant, funny):")

spinner_words = ["Analyzing prompt...", "AI enhancing...", "Reading resources...", "Generating names..."]

if st.button("Generate Names"):
    if pet_type and style:
        spinner_placeholder = st.empty()
        for word in spinner_words:
            with st.spinner(word):
                spinner_placeholder.info(word)
                time.sleep(0.7)
        spinner_placeholder.empty()
        try:
            names = lch.generate_pet_name(pet_type, number_of_names, style)
            if isinstance(names, list):
                st.markdown("**Generated Names:**")
                for name in names:
                    st.write(f"- {name}")
            else:
                # If names is a string, split by comma or newline
                st.markdown("**Generated Names:**")
                for name in str(names).replace('\n', ',').split(','):
                    st.write(f"- {name.strip()}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please fill in all fields.")
