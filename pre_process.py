#Initial Configuration
import json
import nltk
config = './config.json'

#Import configuration file as a JSON object

with open(config, 'r') as config_json:

    config_data = json.load(config_json) 
    columns_of_interest = config_data['columns_of_interest']
    columns_to_vectorize = config_data['columns_to_vectorize']
    nltk.data.path.append(config_data['paths']['nltk_data'])


#Import libraries and modules

import re
import unicodedata
import contractions
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Requirements:
# Download in nltk.data directory:
# NLTK stopwords
# NLTK punkt
# NLTK average_perceptron_tagger


# Auxiliary functions

def pos_tagger(nltk_tag):
    """Map nltk.pos_tag to wordnet Part Of Speech (POS) tag. 

    Args:
        nltk_tag (str): returned tag from nltk.pos_tag function.

    Returns:
        str: POS tag that wordnet.lemmatize() accepts.
    """

    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:       
        return None

class ConsolidatedDataPreProcess():

    """Custom class that import tickets from consolidated database and keep columns of interest only and cast them as string. 
    Creates original_data.pkl
    """

    def __init__(self,df,columns_of_interest = columns_of_interest):

        """
        Object constructor. Defines dataframe keeping columns of interest only. 

        Args:
            df (pandas.DataFrame): tickets dataframe from parquet file in storage
        """

        self.data = df[columns_of_interest].astype(str)
    

    def pre_process(self):
        """
        Data cleaning method: drops duplicated values in 'ticket' column, resets index and fills null values with empty strings.
        """
        self.data.drop_duplicates('ticket', keep = 'last',inplace = True)
        self.data.reset_index(drop=True,inplace = True)
        self.data.fillna(" ", inplace = True)


    def save_to_pickle(self,filepath):
        """
        Data saving method: saves Data Frame as pickle file (.pkl)

        Args:
            filepath (string): path + filename + ".pkl"
        """
        self.data.to_pickle(filepath)


class NLP_ETL():

    """
    Custom class for Natural Language Processing & ETL tasks pipeline:

        - Combine relevant features' columns' text values into a single string.
        - Casing: text conversion into lowercase
        - Contractions expansion
        - Non-alphanumerical characters removal
        - Accents removal
        - Word tokenization
        - Part Of Speech (POS) tagging
        - Lemmatization
        - Stopwords removal
    """

    def __init__(self,df):

        """
        Object constructor.

        Args:
            DataFrame (pandas.DataFrame)
        """
        self.data = df
    
    def combine_features(self,columns_to_vectorize = columns_to_vectorize):

        """
        Method for combining relevant features' columns' text values into a single string.

        Args:
            columns_to_vectorize (list [str]) : list of names of columns containing text to process.  
        """
        self.data['combined'] = ''
        for item in columns_to_vectorize:
            self.data['combined'] = self.data['combined']+self.data[item]+' '
    
    def text_to_minus(self,column_to_process = 'combined'):
        """
        Method for transforming text to lowercase.

        Args:
            column_to_process (str) : name of column which text values will be transformed to lowercase (by default 'combined' column).
        """
        self.data[column_to_process] = self.data[column_to_process].str.lower()

    def expand_contractions(self,text):
        """
        Method for contraction's expansion (e.g., transforms "didn't" into "did not")

        Args:
            text (str) : text with contractions.

        Returns:
            text (str) : text without contractions.
        """
        text = contractions.fix(text)
        return text

    def remove_characters(self, text):
        """
        Method for non-alphanumerical characters removal.

        Args:
            text (str) : text with non-alphanumerical characters.
        Returns:
            text (str) : text without no alphanumeric characters.
        """
        pattern = r'[^a-zA-Z0-9\s]'
        text = re.sub(pattern, '', text)
        return text

    def remove_accented_chars(self, text):
        """
        Method for accented characters transformation and unicode data encoding.

        Args:
            text (str) : text with accented characters.
            
        Returns:
            text (str) : text without accented characters.
        """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = text.encode("utf-8",'ignore')
        return text.decode('ascii')

    def tokenizer(self,text):
        """
        Method for getting word tokens from a given text.

        Args:
            text (str) : text to tokenize.

        Returns:
            (list [str]) : list of word tokens.
        """
        return word_tokenize(text)

    def lemmatizer(self,tokens_list):
        """
        Method for obtaining tokens' lemmas by using Part Of Speech (POS) tags.

        Args:
            tokens_list (list [str]) : list of tokens to lemmatize.

        Returns:
            lemmatized_tokens (list [str]) : list of lemmatized tokens.
        """
        wnl = WordNetLemmatizer()
        pos_tagged_tokens = pos_tag(tokens_list)
        lemmatized_tokens = []
        for word, tag in pos_tagged_tokens:
            if pos_tagger(tag) is None:
                # if there is no available tag, append the token as is
                lemmatized_tokens.append(word)
            else:   
                # else use the tag to lemmatize the token
                lemmatized_tokens.append(wnl.lemmatize(word, pos_tagger(tag)))
        return lemmatized_tokens

    def remove_stopwords(self, lemmatized_tokens):
        """
        Method for stopwords removal

        Args:
            lemmatized_tokens (list [str]): list of tokens with stopwords

        Returns:
            filtered_sentence (list [str]): list of tokens without stopwords
        """

        #Stopwords setting
        stop_words = stopwords.words('english')
        filtered_sentence = [w for w in lemmatized_tokens if not w in stop_words]
        return filtered_sentence
    
    def nlp_pipeline(self,
                     text,
                     expand_contractions = True, 
                     no_char = True, 
                     no_accent = True, 
                     lemmatize = True,
                     no_stopwords = True):
        """
        Method : Natural Language Processing pipeline

        Args:
            text (str): text to transform
            expand_contractions (bool, optional): expands contractions. Defaults to True.
            no_char (bool, optional): removes non-alpanumeric characters. Defaults to True.
            no_accent (bool, optional): transforms accented characters. Defaults to True.
            lemmatize (bool, optional): lemmatize tokens. Defaults to True.
            no_stopwords (bool, optional): removes stopwords. Defaults to True.

        Returns:
            tokens_list (list [str]): list of preprocessed text's tokens
        """
        if expand_contractions:
            text = self.expand_contractions(text)
        if no_char:
            text = self.remove_characters(text)
        if no_accent:
            text = self.remove_accented_chars(text)
        
        tokens_list = self.tokenizer(text)

        if lemmatize:
            tokens_list = self.lemmatizer(tokens_list)

        if no_stopwords:
            tokens_list = self.remove_stopwords(tokens_list)

        return tokens_list


    def nlp_superPipeline(self,columns_to_vectorize,**kwargs):
        """
        Method for feature combining, text lowercasing and subsequent application of NLP pipeline. 

        Args:
            columns_to_vectorize (list [str]) : list of names of columns containing text to process.
        """
        self.combine_features(columns_to_vectorize)
        self.text_to_minus()
        self.data['tokens_list'] = self.data.apply(lambda text: self.nlp_pipeline(text['combined'], **kwargs),axis = 1)

    def combined_features_tokens(self):
        """
        Method for selecting only 'ticket' and 'tokens_list' columns

        Returns:
            Data Frame (pandas.DataFrame): Data Frame containing ticket number and its tokens in two separated columns.
        """
        return self.data[['ticket','tokens_list']].reset_index(drop = True)