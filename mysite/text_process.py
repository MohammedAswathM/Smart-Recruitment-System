import re
import unicodedata
import inflect
import nltk
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from stop_words import get_stop_words

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

def remove_non_ascii(words):
    """Remove non-ASCII characters from the list of tokenized words."""
    return [
        unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        for word in words
    ]

def to_lowercase(words):
    """Convert all characters to lowercase."""
    return [word.lower() for word in words]

def remove_punctuation(words):
    """Remove punctuation."""
    return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']

def replace_numbers(words):
    """Replace integer occurrences with textual representation."""
    p = inflect.engine()
    return [p.number_to_words(word) if word.isdigit() else word for word in words]

def remove_stopwords(words):
    """Remove stop words."""
    stp_words = set(get_stop_words('en'))
    return [word for word in words if word not in stp_words]

def get_keywords(words):
    """Filter only the keywords."""
    keywords_path = 'jobDetails/normalized_keywords.txt'  # Update the path as needed
    try:
        keywords = open(keywords_path, 'r').read().replace('\n', '').split(' ')
        return [word for word in words if word in keywords]
    except FileNotFoundError:
        print(f"Error: Could not find {keywords_path}. Ensure the file exists.")
        return words

def stem_words(words):
    """Stem words."""
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    """Lemmatize verbs."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    """Normalize text."""
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    words = get_keywords(words)
    return words

def generate_questions(job_desc, skill_scores):
    """
    Generate interview questions based on normalized job description and skill scores.
    :param job_desc: Job description as a string.
    :param skill_scores: Dictionary mapping skills to their scores.
    :return: List of questions.
    """
    # Tokenize and normalize job description
    tokens = nltk.word_tokenize(job_desc)
    normalized_keywords = normalize(tokens)

    # Define question templates
    technical_template = "How well do you know {keyword}?"
    project_template = "Can you describe a project where you used {keyword}?"
    challenge_template = "What challenges have you faced when working with {keyword}?"

    # Generate questions based on skill scores
    questions = []
    for keyword in normalized_keywords:
        if keyword in skill_scores:
            score = skill_scores[keyword]
            if score > 7:
                # High-priority skills
                questions.append(technical_template.format(keyword=keyword))
                questions.append(project_template.format(keyword=keyword))
                questions.append(challenge_template.format(keyword=keyword))
            elif 4 <= score <= 7:
                # Medium-priority skills
                questions.append(technical_template.format(keyword=keyword))
                questions.append(project_template.format(keyword=keyword))
            else:
                # Low-priority skills
                questions.append(technical_template.format(keyword=keyword))

    return questions

# Example Usage
if __name__ == "__main__":
    job_description = """
    We are looking for a Python developer with experience in machine learning, 
    data analysis, and web development. Familiarity with Django, Flask, and cloud platforms like AWS is preferred.
    """
    skill_scores = {
        'python': 9,
        'machine': 8,
        'learning': 8,
        'data': 7,
        'analysis': 7,
        'web': 6,
        'development': 6,
        'django': 5,
        'flask': 5,
        'aws': 4
    }

    questions = generate_questions(job_description, skill_scores)

    # Output the generated questions
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
