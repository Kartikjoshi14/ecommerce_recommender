import spacy

nlp = spacy.load("en_core_web_sm")

def process_texts(texts):
    """
    Uses spaCy's nlp.pipe for efficient batch processing.
    Cleans, lemmatizes, removes stopwords and non-alphabetic tokens.
    """
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "parser"]):
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts
