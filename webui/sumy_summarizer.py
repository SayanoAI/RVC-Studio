from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)
print(f"loaded {__name__}")

def get_summary(text: str, num_sentences: int=3):
    
    try:
        results = summarize_text(text,num_sentences,type)

        return results
    except Exception as e:
        print(e)
        return str(e)

def summarize_text(text: str, num_sentences=3, type="text"):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    results = [str(sentence) for sentence in summarizer(parser.document, num_sentences)]

    return " ".join(results)