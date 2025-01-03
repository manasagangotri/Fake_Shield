import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
from googletrans import Translator

st.set_page_config(page_title="News Prediction", page_icon=":earth_africa:")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

translator = Translator()

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        return f"Error in translation: {e}"

def predict_fake(title, text):
    input_str = "<title>" + title + "<content>" + text + "<end>"
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    return dict(zip(["Fake", "Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])]))

def fact_check_with_google(api_key, query):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Unable to fetch results from Google Fact Check API. HTTP {response.status_code}: {response.text}"}

def main():
    st.title("Fake News Prediction")

    # Load Google API key from a secure location or environment variable
    try:
        google_api_key = "AIzaSyAf5v5380xkpo0Rk3kBiSxpxYVBQwcDi2A"
    except KeyError:
        st.error("Google Fact Check API key is missing. Please add it to secrets.")
        return

    # Create the form for user input
    with st.form("news_form"):
        st.subheader("Enter News Details")
        title = st.text_input("Title")
        text = st.text_area("Text")
        language = st.selectbox("Select Language", options=["English", "Other"])
        submit_button = st.form_submit_button("Submit")

    # Process form submission and make prediction
    if submit_button:
        if language == "Other":
            title = translate_to_english(title)
            text = translate_to_english(text)

        prediction = predict_fake(title, text)

        st.subheader("Prediction:")
        st.write("Prediction: ", prediction)

        if prediction.get("Real") > 0.5:
            st.write("This news is predicted to be **real** :muscle:")
        else:
            st.write("This news is predicted to be **fake** :shit:")

        # Fact-check the news using Google Fact Check API
        st.subheader("Fact-Checking Results:")
        query = title if title else text[:100]  # Use title or first 100 chars of text as query
        fact_check_results = fact_check_with_google(google_api_key, query)

        if "error" in fact_check_results:
            st.error(fact_check_results["error"])
        else:
            claims = fact_check_results.get("claims", [])
            if claims:
                for claim in claims:
                    st.write(f"**Claim:** {claim.get('text', 'N/A')}")
                    claim_review = claim.get("claimReview", [])
                    if claim_review:
                        for review in claim_review:
                            st.write(f"- **Publisher:** {review.get('publisher', {}).get('name', 'N/A')}")
                            st.write(f"  **Rating:** {review.get('textualRating', 'N/A')}")
                            st.write(f"  **URL:** [More Info]({review.get('url', '#')})")
            else:
                st.write("No fact-checking information found for this query.")

if __name__ == "__main__":
    main()

