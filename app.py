import streamlit as st
import pandas as pd
import os

# Function to load the CSV file
@st.cache_data
def load_data(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df
    else:
        st.error(f"CSV file '{csv_file}' not found.")
        return None

# Custom CSS for styling
def add_custom_style():
    st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .text-box {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .label {
        font-weight: bold;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .question-answer {
        background-color: #d5f5e3;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    # Add custom styling
    add_custom_style()

    # Add a title and description
    st.markdown('<div class="main-title">Text Document Viewer</div>', unsafe_allow_html=True)
    st.markdown("Select a text document from the dropdown below to view its content in a beautifully formatted way.")

    # Load the CSV file
    csv_file = "data.csv"  # Replace with your CSV file name if different
    df = load_data(csv_file)

    if df is not None:
        # Create dropdown with options "Text Document 1" to "Text Document 100"
        num_docs = min(len(df), 100)
        doc_options = [f"Text Document {i+1}" for i in range(num_docs)]
        selected_doc = st.selectbox("Choose a Text Document:", doc_options)

        # Get the index of the selected document
        doc_index = int(selected_doc.split()[-1]) - 1

        # Display the selected document's content
        st.markdown(f'<div class="subheader">{selected_doc}</div>', unsafe_allow_html=True)
        
        # Create two columns for original and English text
        col1, col2 = st.columns(2)

        # Column 1: Original Text and Summary
        with col1:
            st.markdown('<div class="text-box">', unsafe_allow_html=True)
            st.markdown('<div class="label">Original Text:</div>', unsafe_allow_html=True)
            st.write(df.loc[doc_index, "text"])
            if "summary" in df.columns:
                st.markdown('<div class="label">Original Summary:</div>', unsafe_allow_html=True)
                st.write(df.loc[doc_index, "summary"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Column 2: English Text and Summary
        with col2:
            st.markdown('<div class="text-box">', unsafe_allow_html=True)
            if "text_en" in df.columns:
                st.markdown('<div class="label">English Text:</div>', unsafe_allow_html=True)
                st.write(df.loc[doc_index, "text_en"])
            if "summary_en" in df.columns:
                st.markdown('<div class="label">English Summary:</div>', unsafe_allow_html=True)
                st.write(df.loc[doc_index, "summary_en"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Display Question, Answer, and Confidence below
        st.markdown('<div class="question-answer">', unsafe_allow_html=True)
        if "question" in df.columns:
            st.markdown('<div class="label">Question:</div>', unsafe_allow_html=True)
            st.write(df.loc[doc_index, "question"])
        if "answer" in df.columns:
            st.markdown('<div class="label">Answer:</div>', unsafe_allow_html=True)
            st.write(df.loc[doc_index, "answer"])
        if "answer_confidence" in df.columns:
            st.markdown('<div class="label">Confidence Score:</div>', unsafe_allow_html=True)
            confidence = df.loc[doc_index, "answer_confidence"]
            st.write(f"{confidence:.2f}" if pd.notna(confidence) else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()