from automatic_keyword_generator import *
import os
import re
import ast
import json
import requests

import pdb

import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
import math

from multiprocessing import Pool
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()


def get_keys(text, ngram=1, ntop=10, generator="Spacy"):
    """ Function to extract keywords from a text using a chosen generator

        :param text: The text from which keywords are to be extracted
        :type text: `str`
        :param ngram: No of words used for Ngram
        :type ngram: `int`
        :param ntop: The no of top keywords to be extracted
        :type ntop: `int`
        :param generator: The algorithm to be used for keyword extraction
        :type generator: `str`
        
        :return: List of Keywords
        :rtype: `List`
        """
    # Intialize keyword generator with the text.
    kw = Keyword_generator(text)

    # Extract keywords depending on the type of generator selected - default
    # is Spacy
    if generator == "Spacy":
        return kw.Spacy()

    elif generator == "BERT":
        return kw.BERT(ngram, ntop)

    elif generator == "gensim":
        return kw.Gensim()

    elif generator == "Yake":
        return kw.Yake(ngram, ntop)

    elif generator == "Rake":
        return kw.Rake()
    else:
        return None


class PreProcessing():
    """Class which is equipped with all sorts of Preprocessing & Cleaning techniques"""

    def __init__(self, text):
        """ Constructor 

        :param text: The text which needs to be processed
        :type text: `str`
        
        :return: None
        
        """
        self.text = text

    def remove_urls(self):
        """ Remove URLs from text 
        
        :param None:

        :return: Text after removing URLS
        :rtype: `str`
        """

        self.new_text = ' '.join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\\w+:\\/\\/\\S+)",
                " ",
                self.new_text).split())
        return self.new_text

    def text_lowercase(self):
        """
        Function to convert all alphabets in a text to lowercase
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """
        self.new_text = self.text.lower()
        return self.new_text

    def remove_numbers(self):
        """
        Function to remove numbers in a text
        
        :param None:

        :return: Processed text
        :rtype: `str`
        """

#        self.new_text = re.sub(r'\d+', '', self.new_text)
        self.new_text = re.sub('[^A-Za-z0-9]+', '', self.new_text)
        return self.new_text

    def remove_punctuation(self):
        """
        Function to remove punctuations from the text
        
        :param None:

        :return: Processed text
        :rtype: `str`
        """

        translator = str.maketrans('', '', string.punctuation)
        self.new_text = self.new_text.translate(translator)
        return self.new_text

    def tokenize(self):
        """
        Function to tokenize phrases and tokens
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """

        self.new_text = word_tokenize(self.new_text)
        return self.new_text

    def remove_stopwords(self):
        """
        Function to remove stopwords from the text
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """

        self.new_text = [i for i in self.new_text if i not in stop_words]
        return self.new_text

    def lemmatize(self):
        """
        Function to extract root words - Lemmatizing
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """

        self.new_text = [lemmatizer.lemmatize(
            token) for token in self.new_text]
        return self.new_text

    def stemming(text):
        """
        Function to extract root words - Stemming
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """
        self.new_text = [porter.stem(token) for token in self.new_text]
        return self.new_text

    def remove_letters(self, size):
        """
        Function to remove words from a text with less than n letters
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """

        self.new_text = [i for i in self.new_text if len(i) > size]
        return self.new_text

    def remove_characters(self):
        """
        Function to remove special characters
        
        :param None:
        
        :return: Processed text
        :rtype: `str`
        """
        self.new_text = re.sub('[^A-Za-z0-9]+', ' ', self.new_text)
        return self.new_text


def tokenize(phrase, k=3):
    """ Function which preprocess and tokenize a given phrase 

        :param phrase: Phrase to be tokenized
        :type phrase: `str`
        :param k: Minimum length for a words for it to be retained in the text
        :type k: `str`
        
        :return: Processed tokens
        :rtype: `List`
    """

    preprocess = PreProcessing(phrase)

    preprocess.text_lowercase()
    preprocess.remove_characters()
    preprocess.tokenize()
    preprocess.remove_stopwords()
    preprocess.remove_letters(k)
    processed_tokens = preprocess.lemmatize()

    return processed_tokens


def get_datetime(date_str, year=True):
    """ Converts string Datetime to Datetime Object 

        :param date_str: Datetime in String
        :type date_str: `str`
        :param year: If True, extracts returns the year
        :type year: `bool`
        
        :return: Datetime object
        :rtype: `Datetime`
    """

    if not pd.isna(date_str):
        dt = datetime.strptime(date_str, '%a %b %d %H:%M:%S %Z %Y').date()
        if year:
            return dt.year
        else:
            return dt
    else:
        return np.nan


def parallelize(n_cores, func, arg1):
    """ Function to Parallelize the task on multiple CPU thread

        :param n_cores: No of cores of CPU to be used
        :type n_cores: `int`
        :param func: The function which needs to be parallelized
        :type func: Function()
        :param arg1: List[list of elements, len(list of elements)]
        :type arg1: `str`
        
        :return: List containing the results of the function applied on each element in arg1[0]
        :rtype: `List`
    """

    data_list = []

    with Pool(processes=n_cores) as spool:

        for d in spool.starmap(func, tqdm(arg1, total=len(arg1))):
            data_list.append(d)
            pass
    spool.close()
    spool.join()

    return data_list


def get_request(url, headers, data=''):
    """ Retrieves a webpage with the desired header and payload data and returns the text data
    
        :param url: URL from which data needs to be extracted
        :type url: `str`
        :param headers: headers for the url
        :type headers: `List`
        :param data: Payload data
        :type data: `Dict`
        
        :return: Response from the webpage in text
        :rtype: `Str`
    """

    response = requests.request("GET", url, headers=headers, data=data)
    return response.text


def counter_cosine_similarity(user_id, counterA, counterB):
    """ Calculate the counter cosine similarity for each user_id 
    
        :param user_id: User ID
        :type user_id: `str`
        :param counterA: Keyword list 1
        :type counterA: `List`
        :param counterB: Keyword list 2
        :type counterB: `List`
        
        :return: Dictionary of {User ID: Counter_cosine value}
        :rtype: `Dictionary`
        
    """

    try:

        c1 = Counter(counterA.split(" "))
        c2 = Counter(counterB)

        terms = set(c1).union(c2)
        dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
        magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
        magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
        return {user_id: dotprod / (magA * magB) * 100}
    except BaseException:
        return {user_id: 0}


def save_pandas_to_csv(df, output_path, index):
    """ Saves the dataset to CSV file  
    
        :param output_path: Path where the file needs to be saved
        :type output_path: `str`
        :param index: Whether index should be included while saving
        :type index: `bool`
        
        :return: None
        
    """

    df.to_csv(output_path, index=index)


def get_formatted_date(data, format_='%m%d%Y'):
    """ Function to format date in a required format
    
        :param data: List of dates as string
        :type data: `List`
        :param format_: Format in which date should be returned
        :type format_: `str`
        
        :return: Formatted date
        :rtype: `Datetime`
        
    """
    return data.apply(
        lambda x: dt.datetime.strptime(
            x, format_).date() if bool(
            x.strip()) else dt.date.today())


def merge_databases(dset1, dset2, on, how="inner"):
    """Function to merge two_datasets with key as 'on'
    

        :param dset1: Dataset 1
        :type dset1: `Pandas.DataFrame`
        :param dset2: Dataset 2
        :type dset2: `Pandas.DataFrame`
        :param on: Column on which datasets are to be merged
        :type on: `str`
        :param how: Type of join 
        :type how: `str`
        
        :return: Merged datafram
        :rtype: `Pandas.DataFrame`
        
    """
    return pd.merge(dset1, dset2, on=on, how=how)


def extract_json(base_url, end_url, page):
    """ Extract data from webpage by appending Base_url+page+End_url
    

        :param base_url: Base URL 
        :type base_url: `str`
        :param end_url: End URL
        :type end_url: `str`
        :param page: The page number 
        :type page: `str`
        
        :return: Response from the webpage
        :rtype: `Dict`
        
    """

    url = base_url + str(page) + end_url
    payload = {}
    headers = {
        'accept': 'application/json, text/plain, */*'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    response_str = response.text
    response_dict = json.loads(response_str)

    return response_dict


def create_tokens(df, func, column_name, n_cores):
    """ Function to create tokens for a given column name 

        :param df: DataFrame 
        :type df: `Pandas.DataFrame`
        :param func: Function to be applied
        :param column_name: The name of the column
        :type column_name: `str`
        :param n_cores: No of CPU cores to be used
        :type n_cores: `int`
        
        :return: List containing the results of the function applied on each element of the column
        :rtype: `List`
    """

    list_ = [(i, j)
             for i, j in zip(df[column_name].tolist(), df[column_name].index)]
    # func(list_[0][0], list_[0][1])
    data_list = parallelize(n_cores=n_cores, func=func, arg1=list_)
    return data_list
