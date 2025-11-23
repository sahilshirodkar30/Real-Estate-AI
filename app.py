import streamlit as st
from rag import process_urls, generate_answer

st.title("🏙️ Real Estate Research Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != ""]
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid URL")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("Ask your question:")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(f"- {source}")
    except RuntimeError:
        placeholder.text("⚠️ You must process URLs first before asking questions.")