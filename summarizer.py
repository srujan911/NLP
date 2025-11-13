import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter

# Load the spaCy English model once
NLP = spacy.load('en_core_web_sm')

def extractive_summarizer(text, nlp_model, num_sentences=3):
    """
    Summarizes a given text using an extractive approach.

    Args:
        text (str): The input text to be summarized.
        nlp_model: The loaded spaCy model.
        num_sentences (int): The number of sentences in the desired summary.

    Returns:
        str: The summarized text.
    """
    if not text.strip():
        return ""

    # Process the text with spaCy
    doc = nlp_model(text)

    # Filter out stopwords and punctuation to get the keywords
    # Using lowercased tokens for case-insensitive matching
    keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Calculate the frequency of each keyword
    word_frequencies = Counter(keywords)

    # Normalize the frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # Score sentences based on the frequency of the words they contain
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Get the top N sentences with the highest scores
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # Select the top 'num_sentences' sentences
    final_sentences = [sent.text for sent in summarized_sentences[:num_sentences]]

    # Join the sentences to form the final summary
    summary = " ".join(final_sentences)
    return summary

# --- Example Usage ---
if __name__ == "__main__":
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.
    """

    summary = extractive_summarizer(sample_text, NLP, 2)
    print("Original Text:")
    print(sample_text)
    print("\nSummarized Text:")
    print(summary)
