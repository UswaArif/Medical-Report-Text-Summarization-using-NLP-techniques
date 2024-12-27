from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
import spacy
import nltk

#tokenzier
nltk.download('punkt')
#database
nltk.download('wordnet')
#translate words from WordNet into various languages.
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")

#medical abbreviations dictionary
medical_abbreviations = {
    "BP": "blood pressure",
    "HR": "heart rate",
    "O2": "oxygen",
    "ECG": "electrocardiogram"
}

app = Flask(__name__)
CORS(app)

# Load pre-trained BART model and tokenizer from Hugging Face
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_text(text):
    """
    Preprocesses the medical report text.
    
    - Expands abbreviations
    - Tokenizes the text
    - Removes stopwords
    - Lemmatizes words
    
    Parameters:
    - text (str): The raw medical report text.
    
    Returns:
    - str: The preprocessed text.
    """
    #Check medical abbreviations
    for abbr, full_form in medical_abbreviations.items():
        text = text.replace(abbr, full_form)

    doc = nlp(text)
    
    #Remove stopwords, punctuation, and lemmatize the tokens
    processed_tokens = [
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    ]
    
    preprocessed_text = " ".join(processed_tokens)
    
    return preprocessed_text

def summarize_medical_report(text, max_input_length=1024, max_output_length=150):
    """
    Summarizes a medical report using Hugging Face BART model after preprocessing.
    
    Parameters:
    - text (str): The medical report text to summarize.
    - max_input_length (int): The maximum input token length for the model.
    - max_output_length (int): The maximum output summary token length.
    
    Returns:
    - str: The summarized medical report.
    """
    preprocessed_text = preprocess_text(text)
    
    inputs = tokenizer(preprocessed_text, max_length=max_input_length, truncation=True, return_tensors="pt", padding=True)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output_length,
        num_beams=4,  
        length_penalty=2.0,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Endpoint to summarize the medical report with text preprocessing.
    """
    data = request.get_json()
    report = data.get("report", "")
    if not report:
        return jsonify({"error": "No report provided"}), 400
    
    summary = summarize_medical_report(report)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
