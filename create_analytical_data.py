import os
import re
import ast
import json
import yaml

import argparse
import requests

from datetime import datetime
import pandas as pd
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

from helpers import merge_databases, save_pandas_to_csv, parallelize, get_datetime, tokenize, create_tokens, get_keys
from automatic_keyword_generator import *

import pdb

extra_stopwords = []


def user_keywords(user_key, i):
    """ Function to calculate tokens from user's keywords 

        :param user_key: Text from which kerwords are to be extracted
        :type user_key: `str`
        :param i: User ID (dummy_variable)
        :type i: `str`
        
        :return: Space separated set of keywords
        :rtype: `str`
    """
    
    try:
        key_text = " ".join(user_key.split("||"))
        user_keys = " ".join(tokenize(key_text))

    except BaseException:
        user_keys = ""

    return user_keys


def user_o_keywords(user_overview, i):
    """ Function to calculate tokens from user's Overview 

        :param user_overview: Text from which kerwords are to be extracted
        :type user_overview: `str`
        :param i: User ID (dummy_variable)
        :type i: `str`
        
        :return: Space separated set of keywords
        :rtype: `str`
    """

    try:
        user_keys = " ".join(tokenize(user_overview))
    except BaseException:
        user_keys = ""

    return user_keys


def get_author_pubinfo(scholar_df, i, top_n=5, top_title=True):
    """ Function to extract information of top N publications of the author 

        :param scholar_df: DataFrame containing user's all information
        :type scholar_df: `Pandas.DataFrame`
        :param i: User ID (dummy_variable)
        :type i: `str`
        :param top_n: Based on relevancy, the number of top Titles will be used
        :type top_n: `int`
        :param top_title: If True, only top N pulications will be extracted. Else all publicatio data will be used.
        :type top_title: `bool`
        
        :return: Tuple of Dictionaries. Each distionary contain User_id as key and keyworks from Publication title / User keywords as values
        :rtype: `Tuple`
    """

    user = scholar_df["user_id"].values[0]

    try:
        if top_title:
            title = " ".join(scholar_df["title"][:top_n])
        else:
            title = " ".join(scholar_df["title"])

        title_keys = " ".join(list(set([i for i in get_keys(
            text=title, generator="Spacy", ntop=top_n) if len(i) > 3])))

        keys = [ast.literal_eval(
            i) for i in scholar_df["keywords"].values if not pd.isna(i)]
        merge_keys = [i.split(" ") for k in keys for i in k]
        word_keys = " ".join(
            list(set([i for k in merge_keys for i in k if len(i) > 3])))

        return {user: title_keys}, {user: word_keys}

    except BaseException:
        return {user: ""}, {user: ""}
    

def get_author_year_pubinfo(scholar_df, i, years=5, top_title=True):
    """ Function to extract information of publications based on published year by the author 

        :param scholar_df: DataFrame containing user's all information
        :type scholar_df: `Pandas.DataFrame`
        :param i: User ID (dummy_variable)
        :type i: `str`
        :param year: # of years the publications should be extracted
        :type top_n: `int`
        :param top_title: If True, only top N pulications will be extracted. Else all publicatio data will be used.
        :type top_title: `bool`
        
        :return: Tuple of Dictionaries. Each distionary contain User_id as key and keyworks from Publication title / User keywords as values
        :rtype: `Tuple`
    """

    user = scholar_df["user_id"].values[0]
    
    try:
        texts_ = []
        for i in range(1, years+1):
            sub_df = scholar_df[scholar_df['publication_year']>= datetime.today().year-i]
            if top_title:
                title = " ".join(sub_df['title'])

            else:
                title = " "

            title_keys = " ".join(list(set([i for i in get_keys(title ,generator="Spacy", ntop = 10) if len(i)>3])))

            keys = [ast.literal_eval(i) for i in sub_df["keywords"].values if not pd.isna(i)]
            merge_keys = [i.split(" ") for k in keys for i in k]
            word_keys = " ".join(list(set([i for k in merge_keys for i in k if len(i) > 3])))
            
            pub_text = " ".join([title_keys, word_keys])
            
            texts_.append(pub_text)
        return {user:texts_}

    except:
        return {user:[""]*years}


def user_org_keywords(user_organisation, i):
    """ Function to calculate tokens from user's organization 

        :param user_organisation: Text from which kerwords are to be extracted
        :type user_organisation: `str`
        :param i: User ID (dummy_variable)
        :type i: `str`
        
        :return: Space separated set of keywords
        :rtype: `str`
    """
    #TODO - STOPWORDS are not used correctly

    try:
        user_keys = " ".join([i for i in tokenize(
            user_organisation) if i not in extra_stopwords])
    except BaseException:
        user_keys = ""

    return user_keys

class Analytical_Data_Creator():
    """ Class which will create a user (read scholar) database with details from his profile page and relevant publications. 

        :param user_organisation: Text from which kerwords are to be extracted
        :type user_organisation: `str`
        :param i: User ID (dummy_variable)
        :type i: `str`
        
        :return: Space separated set of keywords
        :rtype: `str`
    """

    def __init__(self, n_cores, univ_name, params):
        """ Constructor

        :param n_cores: No: of CPU cores to be utilized
        :type n_cores: `int`
        :param univ_name: Name of the scholar's university
        :type univ_name: `str`
        :param params: Dictionary of default values for each parameter as read from the CONFIG.yml file
        :type params: `Dict`
        
        :return: None
        """

        # Initialize the parameters
        self.n_cores = params['CPU_COUNT'] if n_cores == 0 else n_cores
        self.output_path = params['OUTPUT_PATH']
        self.analytical_filename = params["ANALYTICAL_DATASET"]
        self.year_data_filename = params["YEAR_WISE_PUBLICATION_DATASET"]
        self.user_df = pd.read_csv(
            os.path.join(
                self.output_path,
                params["SCHOLARS_DATASET"]))
        self.pub_df = pd.read_csv(
            os.path.join(
                self.output_path,
                params["PUBLICATION_DATASET"]))
        global extra_stopwords
        extra_stopwords = params['UNIV_DETAILS'][args.univ_name]['STOPWORDS']

    def create_user_token_data(self):
        """ Function to create tokens from Profile page  (Organization, Overview and Keyword sections) for all users 

        :param None: 
        
        :return: None
        """

        user_tokens = create_tokens(
            func=user_keywords,
            column_name="Keywords",
            df=self.user_df,
            n_cores=self.n_cores)
        overview_tokens = create_tokens(
            func=user_o_keywords,
            column_name="Overview",
            df=self.user_df,
            n_cores=self.n_cores)
        org_tokens = create_tokens(
            func=user_org_keywords,
            column_name="Organizations",
            df=self.user_df,
            n_cores=self.n_cores)

        self.sub_df = self.user_df[["User_id", "Netid", "Email", "Name"]]
        self.sub_df["Keywords"] = user_tokens
        self.sub_df["Overview"] = overview_tokens
        self.sub_df["Organization"] = org_tokens
        self.sub_df.rename(columns={"User_id": "user_id"}, inplace=True)

    def create_publication_data(self):
        """ Main function to process and compile the final Scholars' dataset 

        :param None: 
        
        :return: None
        """

        self.pub_df['publication_year'] = [
            get_datetime(
                self.pub_df["publicationDate"][i]) for i in range(
                0, self.pub_df.shape[0])]
        self.pub_df['publication_year'].fillna(0, inplace=True)
        self.pub_df['publication_dt'] = [
            get_datetime(
                self.pub_df["publicationDate"][i],
                False) for i in range(
                0,
                self.pub_df.shape[0])]

        article_data = self.pub_df.groupby("user_id").apply(
            lambda x: x.sort_values('publication_dt', ascending=False))
        article_data = article_data[["user_id",
                                     "publication_dt", "publication_year", "title", "keywords"]]
        article_data['publication_year'] = article_data['publication_year'].astype('int')

        split_data = []
        for i in self.pub_df["user_id"].unique():
            try:
                split_data.append(article_data.loc[i])
            except BaseException:
                continue

        pub_list = [(i, j) for i, j in zip(split_data, range(len(split_data)))]
        pub_tokens = []
        pub_tokens = parallelize(
            n_cores=self.n_cores,
            func=get_author_pubinfo,
            arg1=pub_list)

        pub_title_list = [i[0] for i in pub_tokens]
        pub_key_list = [i[1] for i in pub_tokens]

        key_df = pd.DataFrame({"user_id": [list(i.keys())[0] for i in pub_key_list], "pub_keyword": [
                              list(i.values())[0] for i in pub_key_list]})
        title_df = pd.DataFrame({"user_id": [list(i.keys())[0] for i in pub_title_list], "pub_title": [
                                list(i.values())[0] for i in pub_title_list]})

        self.pub_info = pd.merge(key_df, title_df, on="user_id", how="inner")
        self.ad = merge_databases(
            dset1=self.sub_df,
            dset2=self.pub_info,
            on="user_id",
            how="inner")

        save_pandas_to_csv(
            df=self.ad,
            output_path=os.path.join(
                self.output_path,
                self.analytical_filename),
            index=False)
        
        year_pub_tokens = []
        year_pub_tokens = parallelize(
                    n_cores=self.n_cores,
                    func=get_author_year_pubinfo,
                    arg1=pub_list)
        
        pub_year_df = pd.concat([pd.DataFrame(d.values(), index=d.keys(), columns=["pub_text_"+str(i)+"_year" for i in range(1, len(list(d.values())[0])+1)]) for d in year_pub_tokens])
        pub_year_df.reset_index(inplace=True)
        pub_year_df.rename(columns={'index':'user_id'}, inplace=True)
        
        self.pub_year_ad = merge_databases(
        dset1=self.sub_df,
        dset2=pub_year_df,
        on="user_id",
        how="inner")
        
        save_pandas_to_csv(
        df=self.pub_year_ad,
        output_path=os.path.join(
            self.output_path,
            self.year_data_filename),
        index=False)

if __name__ == '__main__':

    # Read arguments from command line (cmd). If no input via cmd, use config
    # file
    parser = argparse.ArgumentParser(description="Parameter file")
    parser.add_argument(
        '--config_file',
        metavar='FILENAME',
        type=str,
        default='config.yml',
        help='Parameter file name in yaml format')
    parser.add_argument(
        '--univ_name',
        metavar='UNIV_NAME',
        type=str,
        default='TAMU',
        choices=[
            'TAMU',
            'UFL'],
        help='NAME of University')
    parser.add_argument(
        '--n_cores',
        metavar='CPU_COUNT',
        type=int,
        default=0,
        help='No of CPU threads to be used')
    args = parser.parse_args()

    try:
        params = yaml.safe_load(open(args.config_file))
    except BaseException:
        print(f'Error loading parameter file: {args.config_file}.')
        sys.exit(1)

    analytical_data_creator = Analytical_Data_Creator(
        n_cores=args.n_cores, univ_name=args.univ_name, params=params)
    analytical_data_creator.create_user_token_data()
    analytical_data_creator.create_publication_data()