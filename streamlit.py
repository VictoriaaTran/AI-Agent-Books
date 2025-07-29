import streamlit as st
from query_input import get_context, generate_prompt, get_response

st.title("Lord of the Rings QA 📚")
st.write("Lord of the Rings book repository contains 4 books: The Hobbit |\nThe Fellowship of the Ring |\nThe Return of the King |\n and The Two Towers.")
st.write("Note: The model is trained on the 4 books and may not answer questions about other topics. If no answer is found, rephrase the question for better results.")
question = st.text_input("Enter Question:")

if question.strip():
    with st.spinner("Searching..."):
        context_text, result = get_context(question)
        prompt = generate_prompt(question, context_text)
        response, sources = get_response(prompt, result)
        st.write("#### Answer:")
        st.write("Sources: ", sources if sources else "No sources found.")
        st.write(response)

        
