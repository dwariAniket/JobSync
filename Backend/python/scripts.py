import sys
import requests
from io import BytesIO
from pypdf import PdfReader
import json
import nltk
import re
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def fix_spaced_text(text):
    """
    Remove spaces that were added between each character during PDF extraction
    """
    # Remove spaces between letters but keep spaces between words
    fixed_text = re.sub(r'(?<=\w) (?=\w)', '', text)
    return fixed_text

def main():
    try:
        # Get command line arguments
        url = sys.argv[1]
        job_description = "ABC"
        
        # Download and parse PDF
        response = requests.get(url)
        response.raise_for_status()
        
        file_like = BytesIO(response.content)
        reader = PdfReader(file_like)
        num_pages = len(reader.pages)
        first_page_text = reader.pages[0].extract_text()
        
        # FIX: Remove spaces between characters
        first_page_text = fix_spaced_text(first_page_text)
        
        # Text processing functions
        port_stem = PorterStemmer()
        
        def stemming(content):
            stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                             if not word in stopwords.words("english")]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content
        
        # Initialize BERT model (once)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Process resume text
        text = first_page_text
        
        # Use NLTK for sentence tokenization
        sentences = nltk.sent_tokenize(text)
        sentences = [s for s in sentences if s.strip()]
        
        # Filter sentences for skills
        skill_keywords = ["skills", "proficient", "technical", "soft skills", 
                         "experience", "worked as", "handled", "knowledge", "expertise"]
        education_keywords = ["education", "bachelor", "diploma", "secondary", 
                             "university", "college", "makaut", "wbscte", "wbbse", 
                             "cgpa", "coursework", "degree"]
        
        filtered_sentences = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            is_skill_related = any(keyword in lower_sentence for keyword in skill_keywords)
            is_education_related = any(keyword in lower_sentence for keyword in education_keywords)
            if is_skill_related and not is_education_related:
                filtered_sentences.append(sentence)
        
        # If no skill sentences found, use all non-education sentences
        if not filtered_sentences:
            filtered_sentences = [s for s in sentences if not any(
                keyword in s.lower() for keyword in education_keywords)]
        
        # Create summary from filtered sentences
        summary = " ".join(filtered_sentences)
        
        # Preprocess texts
        processed_jd = stemming(job_description)
        processed_resume = stemming(summary)
        
        # Get BERT embeddings
        jd_embedding = get_bert_embedding(processed_jd)
        resume_embedding = get_bert_embedding(processed_resume)
        
        # Calculate cosine similarity
        similarity = torch_cosine_similarity(
            jd_embedding.unsqueeze(0), 
            resume_embedding.unsqueeze(0)
        ).item()
        
        # Prepare output
        result = {
            "num_pages": num_pages,
            "summary": summary,
            "similarity_score": similarity,
            "status": "success"
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e)
        }
        print(json.dumps(error_result))
    
    sys.stdout.flush()

if __name__ == "__main__":
    main()