import os
import re
import ast
import sys
import json
import yaml

import requests
import argparse

from datetime import datetime
import pandas as pd
import numpy as np

from helpers import *
from automatic_keyword_generator import *

from collections import Counter
import math

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')


def get_column_names():

    feat_cols = [
        'Keywords',
        'Overview',
        'Organization',
        'pub_keyword',
        'pub_title']
    cfp = ["Query"]
    return [i + "_" + j + "_sim" for i in feat_cols for j in cfp]


class Top_Scholar_Identifier():
    """This is a class to identify the top N scholars for a given proposal. 
    The proposal dataset created using 'main_extractor.py' will be utilized to get details of the proposal / grant. 
    The analytical dataset of user-publications created using 'create_analytical_data.py' will be utilized to get scholar profiles.
    """

    def __init__(self, n_cores, top_k, generator_, query, params):
        """ Constructor

        :param n_cores: No: of CPU cores to be used for the process
        :type n_cores: `int`
        :param id_no: Opportunity Number of the proposal
        :type id_no: `str`
        :param top_k: The number of scholars to be recommended
        :type top_k: `int`
        :param generator_: The generator to be used for keyword extraction
        :type generator_: `str`
        :param agency: The agency which is awarding the grant
        :type agency: `str`      
        :param params: Parameters read from the configuration file
        :type params: `dict`     
        
        """

        # Set the parameters
        self.n_cores = params['CPU_COUNT'] if n_cores == 0 else n_cores
        self.output_path = params['OUTPUT_PATH']
        self.top_k = params['TOP_K_SCHOLARS'] if top_k == 0 else top_k
        self.generator_ = generator_
        self.analytical_filename = params["ANALYTICAL_DATASET"]
        self.scholars_filename = params["SCHOLARS_DATASET"]
        self.query = query
        self.params = params

    def read_data(self):
        """ Function which will read data from the initialized CSV files
        regarding proposal and scholar datails and save them as pandas dataframe
                
        :param None : 

        :return: None
        :rtype: 
        """

        # Read scholars' basic data
        self.user_df = pd.read_csv(
            os.path.join(
                self.output_path,
                self.scholars_filename))

        # Read scholars' publication data
        self.ad = pd.read_csv(
            os.path.join(
                self.output_path,
                self.analytical_filename))

    def get_section_keys_for_proposal(self):
        """ Function to get keywords from Description, Title adn Department sections of the proposal text
        
        :param None : 

        :return: None
        :rtype:   
        """

        # Get keys from the Abstract of proposal
        self.query = [
            i for i in get_keys(
                self.query,
                generator=self.generator_,
                ntop=self.top_k) if len(i) > 3]

    def get_top_scholars(self, ntop_=20):
        """ Main function to calculate the scholars suitable for the given proposal
        
        :param ntop_: No of scholars to be recommended
        :type ntop_: `int`
        
        :return: self.recommend_df
        :rtype: class `Pandas.DataFrame`

        """

        # Create a list of lists. Each sublist contain a user's id, his/her section keys (from analytical database), proposal's sections keys
        # That is if there are 'n' users with 'm' sentions from his profile and 'k' section in proposal, the main list will consist of 'n'*'m'*'k' sublists
        self.similarity_lists = []
        feat_cols = [
            'Keywords',
            'Overview',
            'Organization',
            'pub_keyword',
            'pub_title']

        for i in feat_cols:
            self.similarity_lists.append([(i, j, self.query) for i, j in zip(
                self.ad["user_id"], self.ad[i])])

        # Run counter cosine similarity as parallel tasks
        self.score_lists = []
        for k in self.similarity_lists:
            sims = parallelize(
                self.n_cores,
                func=counter_cosine_similarity,
                arg1=k)
            self.score_lists.append(sims)

        self.sim_df = pd.DataFrame(
            {"user_id": [list(i.keys())[0] for i in sims]})

        col_names = get_column_names()
        for col, k in zip(col_names, range(len(col_names))):
            self.sim_df[col] = [list(i.values())[0]
                                for i in self.score_lists[k]]

        # Append the similarity values to the original dataframe
        if self.params['DEFAULT']:
            self.sim_df["total_sim"] = self.sim_df.sum(axis=1)
        else:
            sub_df = pd.DataFrame()
            for key, value in self.params['KPI_WEIGHTAGE'].items():
                cols = [col for col in self.sim_df.columns if col.startswith(key)]
                sub_df[key] = self.sim_df[cols].sum(axis=1)*value
            self.sim_df['total_sim'] = sub_df[list(self.params['KPI_WEIGHTAGE'].keys())].sum(axis=1).values/100

        print("Max of self.sim[total_sim] :", self.sim_df["total_sim"].max())
        
        self.recommend_df = pd.merge(self.user_df,self.sim_df, left_on='User_id', right_on='user_id', how='left')
        self.recommend_df.sort_values(["total_sim", "n_publications"], ascending=[False, False], inplace=True)
        self.recommend_df = self.recommend_df[:self.top_k]

        return self.recommend_df
    
    def get_similarity(self, user_id ,text1):
            
        try:
            # Encode the two texts into embeddings
            embeddings1 = model.encode([text1], convert_to_tensor=True)
            embeddings2 = model.encode([self.query], convert_to_tensor=True)

            # Calculate cosine similarity between the embeddings
            similarity = cosine_similarity(embeddings1, embeddings2)

            return {'user_id':user_id, 'sim_score':similarity[0][0]}
        except:
            return {'user_id':user_id, 'sim_score': 0}
        
    def calculate_sbert_sim(self):
        
        self.ad[['Keywords','Overview','Organization','pub_keyword','pub_title']] = self.ad[['Keywords','Overview','Organization','pub_keyword','pub_title']].fillna("")

        print(self.ad.isnull().sum())
        self.ad['merged'] = self.ad[[
                    'Keywords',
                    'Overview',
                    'Organization',
                    'pub_keyword',
                    'pub_title']].apply(lambda x: ' '.join(x), axis=1)
            
        sim_list = [(i,j) for i,j in zip(self.ad['user_id'], self.ad['merged'])]

        # Run as parallel tasks
        sims = parallelize(
            self.n_cores,
            func=self.get_similarity,
            arg1=sim_list)
        
        self.sim_df = pd.DataFrame(sims)
        
        self.recommend_df = pd.merge(self.user_df,self.sim_df, left_on='User_id', right_on='user_id', how='left')
        self.recommend_df.sort_values(["sim_score", "n_publications"], ascending=[False, False], inplace=True)
        self.recommend_df = self.recommend_df[:self.top_k]
        
        return self.recommend_df

    

if __name__ == "__main__":

    """ Read arguments from command line (cmd). If no input via cmd, use config
        file 
    """
    parser = argparse.ArgumentParser(description="Parameter file")
    parser.add_argument(
        '--config_file',
        metavar='FILENAME',
        type=str,
        default='config.yml',
        help='Parameter file name in yaml format')
    parser.add_argument(
        '--top_k',
        metavar='TOP_K_SCHOLARS',
        type=int,
        default=0,
        help='Enter an integer K (K>0) to identify the top K scholars')
    parser.add_argument(
        '--generator',
        metavar='KEYWORD_GENERATOR',
        type=str,
        default="Spacy",
        help='Generator for automatic keyword extraction')
    parser.add_argument(
        '--n_cores',
        metavar='CPU_COUNT',
        type=int,
        default=0,
        help='No of CPU threads to be used')
    parser.add_argument(
        '--query',
        metavar='QUERY',
        type=str,
        default='',
        help='User query')
    parser.add_argument(
        '--method',
        metavar='METHOD',
        type=str,
        default='SBERT',
        help='Text similarity Method')
    args = parser.parse_args()
    
    # Read configuration file. If not successfull end the program
    try:
        params = yaml.safe_load(open('config.yml'))
    except BaseException:
        print(f'Error loading parameter file: {args.config_file}.')
        sys.exit(1)

    # Initialize a class object with all parameters
    obj = Top_Scholar_Identifier(
        n_cores=args.n_cores,
        top_k=args.top_k,
        generator_=args.generator,
        query = args.query,
        params=params)

    # Reads (CSV file) with data regarding Proposal, Scholar details and
    obj.read_data()

    # Extract keyword for proposal
    obj.get_section_keys_for_proposal()

    # Get recommendations
    if args.method == 'counter':
        recommendations = obj.get_top_scholars(ntop_=args.top_k)
    else:
        recommendations = obj.calculate_sbert_sim()

    # Save the recommendation
    save_pandas_to_csv(
        df=recommendations,
        output_path=os.path.join(
            obj.output_path,
            params['PROPOSAL_RECOMMENDATIONS_FILENAME']),
        index=False)