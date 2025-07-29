import streamlit as st
from pathlib import Path
from query_input import get_context, generate_prompt, get_response

st.title("Lord of the Rings QA 📚")
st.write("Lord of the Rings book repository contains 4 books: The Hobbit |\nThe Fellowship of the Ring |\nThe Return of the King |\n and The Two Towers.")
st.write("Note: The model is trained on the 4 books and may not answer questions about other topics. If no answer is found, rephrase the question for better results.")
question = st.text_input("Enter Question:")

if question.strip():
    with st.spinner("Searching..."):
        try:
            context_text, result = get_context(question)
            if not context_text:
                st.write("No relevant context found. Please rephrase your question.")
            else:
                prompt = generate_prompt(question, context_text)
                response, sources = get_response(prompt, result)
                st.write("#### Answer:")
                st.write(response)

                if sources:
                    source_name = Path(sources[0]).stem
                    st.write("Sources: ", source_name)
                else:
                    st.write("Sources: No sources found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
