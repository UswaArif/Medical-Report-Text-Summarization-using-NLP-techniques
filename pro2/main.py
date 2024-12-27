from flask import Flask, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS
import spacy
import nltk
import pandas as pd
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")

#Medical abbreviations 
medical_abbreviations = {
    "BP": "blood pressure",
    "HR": "heart rate",
    "O2": "oxygen",
    "ECG": "electrocardiogram"
}

app = Flask(__name__)
CORS(app)

#Load pre-trained BART model and tokenizer from Hugging Face
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_text(text):
    """
    Preprocesses the medical report text:
    - Expands abbreviations
    - Removes stopwords, punctuation, and lemmatizes tokens
    """
    for abbr, full_form in medical_abbreviations.items():
        text = text.replace(abbr, full_form)
    doc = nlp(text)
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    preprocessed_text = " ".join(processed_tokens)
    return preprocessed_text

def summarize_medical_report(text, max_input_length=1024, max_output_length=150):
    """
    Summarizes a medical report after preprocessing.
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
def summarize_reports():
    """
    Endpoint to accept Excel file, summarize each report, and return a new Excel file with summaries.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "File format not supported. Please upload an Excel file."}), 400

    df = pd.read_excel(file)

    required_columns = ['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition', 
                        'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 
                        'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date', 
                        'Medication', 'Test Results']
    
    for column in required_columns:
        if column not in df.columns:
            return jsonify({"error": f"Missing column: {column}"}), 400

    df['Medical Condition Summary'] = df['Medical Condition'].apply(summarize_medical_report)
    df['Test Results Summary'] = df['Test Results'].apply(summarize_medical_report)

    output_file = 'summarized_medical_reports.xlsx'
    df.to_excel(output_file, index=False)

    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
