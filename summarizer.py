import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
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
    doc = nlp_model(text)
    keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    word_frequencies = Counter(keywords)
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    final_sentences = [sent.text for sent in summarized_sentences[:num_sentences]]
    summary = " ".join(final_sentences)
    return summary
if __name__ == "__main__":
    user_text = input("Please enter the text you would like to summarize:\n")

    summary = extractive_summarizer(user_text, NLP, 2)
    print("\nSummarized Text:")
    print(summary)
