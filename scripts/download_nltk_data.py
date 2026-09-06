"""One-off NLTK data downloads required by Unstructured's text parsing."""
import nltk

if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger")
