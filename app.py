import streamlit as st
import pickle
import re
import nltk

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# Load the models
clf = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text.strip()

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        # 1. Clean the text
        cleaned_resume = clean_resume(resume_text)
        
        # 2. Transform using TF-IDF
        input_features = tfidf.transform([cleaned_resume])
        
        # 3. Predict
        prediction_id = clf.predict(input_features)[0]
        
        # 4. Map ID to Category (Keep this INSIDE the if block)
        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
            20: "Python Developer", 24: "Web Designing", 12: "HR",
            13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales",
            16: "Mechanical Engineer", 1: "Arts", 7: "Database",
            11: "Electrical Engineering", 14: "Health and fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
            2: "Automation Testing", 17: "Network Security Engineer",
            21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # 5. Display Result
        st.success(f"Predicted Category: {category_name}")

if __name__ == "__main__":
    main()