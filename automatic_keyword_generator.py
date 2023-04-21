from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from collections import Counter
from rake_nltk import Rake
import yake
import spacy
import re
import warnings
warnings.filterwarnings("ignore")


# from gensim.summarization import keywords


rake_nltk_var = Rake()
stop_words = "english"
nlp = spacy.load('en_core_web_md')
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def countVectorizer(n_gram, text):
    count = CountVectorizer(
        ngram_range=n_gram,
        stop_words=stop_words).fit(text)
    candidates = count.get_feature_names()

    return candidates


class Keyword_generator():
    """ Class containing various algorithms to generate keywords.
        Algorithms include Yake, Gensim, Rake, Bert, Spacy.
        """
    def __init__(self, text):
        """ Constructor
        
        :param text: the text to be used to extract keywords from
        :type text: `str'

        :return: None
        """
        self.text = text

    def gensim(self):
        """ Function containing Gensim Algorithm to extract keywords
        
        :param None: 
        
        :return: List of extracted keywords
		:rtype: `List`

        """

        return keywords(self.text).split('\n')

    def Yake(
            self,
            max_ngram_size,
            numOfKeywords,
            language="en",
            deduplication_threshold=0.9):
        """ Function containing YAKE algorithm to extract keywords

        :param max_ngram_size: Int based on word grams
        :type max_ngram_size: `int`
        :param numOfKeywords: Ordered on relevancy, the number of top keywords to be returned
        :type numOfKeywords: `int`
        :param language: Language of the text (default = `en`)
        :type language: `int`
        :param deduplication_threshold: Duplication of words in keywords
        :type deduplication_threshold: `float`
        
        :return: List of extracted keywords
        :rtype: `List`
        """
        
        custom_kw_extractor = yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=deduplication_threshold,
            top=numOfKeywords,
            features=None)

        keywords = custom_kw_extractor.extract_keywords(self.text)

        return [i[0] for i in keywords]

    def Rake(self):
        """ Function containing RAKE algorithm to extract keywords

        :param None:
        
        :return: List of extracted keywords
        :rtype: `List`
        """

        rake_nltk_var.extract_keywords_from_text(self.text)
        keyword_extracted = rake_nltk_var.get_ranked_phrases()

        modified_keys = [re.sub('[^A-Za-z0-9]+', ' ', i)
                         for i in keyword_extracted]
        modified_keys = [''.join([i for i in k if not i.isdigit()])
                         for k in modified_keys]
        output = [i for i in modified_keys if len(i) > 1]

        return output

    def BERT(self, n_gram=1, top_n=5):
        """ Function containing BERT algorithm to extract keywords

        :param n_gram: No of continuous sequence of words to be used 
        :type n_gram: `int`
        :param top_n:  Ordered on relevancy, the number of top keywords to be returned
        :type top_n: `int`
        
        :return: List of extracted keywords
        :rtype: `List`
        """

        stop_words = "english"

        count = CountVectorizer(ngram_range=(
            n_gram, n_gram), stop_words=stop_words).fit([self.text])
        candidates = count.get_feature_names()

        doc_embedding = model.encode([self.text])
        candidate_embeddings = model.encode(candidates)
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        keywords = [candidates[index]
                    for index in distances.argsort()[0][-top_n:]]

        return keywords

    def Spacy(self):
        """ Function containing Spacy algorithm to extract keywords

        :param None: 

        :return: List of extracted keywords
        :rtype: `List`
        """

        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN']
        doc = nlp(self.text.lower())
        for token in doc:

            if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue

            if (token.pos_ in pos_tag):
                result.append(token.text)

        return result
