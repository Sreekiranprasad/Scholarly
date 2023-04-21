import os
import re
import ast
import json
import yaml
import sys

import requests
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
import pdb

from helpers import parallelize
from helpers import get_request


class Extract_Publications():
    """Class which will extract all the publication details of all the scholars of a given university
    """
    
    def __init__(self, n_cores, univ_name, params):
        """ Function to retrieve response from NIH page. The processed response will be save as self.soup

        :param n_cores: No: of CPU cores to be utilized
        :type n_cores: `int`
        :param univ_name: Name of the scholar's university
        :type univ_name: `str`
        :param params: Dictionary of default values for each parameter as read from the CONFIG.yml file
        :type params: `Dict`
        
        :return: None
        """

        # Set the parameters
        self.n_cores = params['CPU_COUNT'] if n_cores == 0 else n_cores
        self.output_path = params['OUTPUT_PATH']
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.output_path )
        self.publication_file_name = params['PUBLICATION_DATASET']
        self.user_df = pd.read_csv(
            os.path.join(
                params['OUTPUT_PATH'],
                params["SCHOLARS_DATASET"]))
        self.univ_details = params['UNIV_DETAILS'][univ_name]
        self.pub_data = []

    def get_publication_ids(self, user_id, str_):
        """ Returns a dictionary where each user_id is a key and a list of his/her publications as values

        :param user_id: User_id for each user whose publications are to be extracted
        :type user_id: `str`
        param str_: WIP
        :type str_: `str`
        
        :return: Dictonary of {User IDs : List of publications}
        :rtype: class `Dictionary `
        """
        
        publications_ids = []
        user_dict = {}

        try:
            dict_list = ast.literal_eval(str_)

            for d_ in dict_list:
                publications_ids.append(d_["id"])
            user_dict[user_id] = publications_ids

            return user_dict

        except BaseException:
            user_dict[user_id] = publications_ids
            return user_dict

    def create_user_publication_data(self, user_id, pub_ids):
        """ Creates the publication data for a single user by scraping university webpage.
        The function takes in a publication ID and get complete detail of the data from University webpage.

        :param user_id: User_id for each user whose publications are to be extracted
        :type user_id: `str`
        :param pub_ids: List of all publication IDs for the user
        :type pub_ids: `List`
        
        :return: Dataframe for each user where each row is a publication of a user. Total no of rows = n_publications
        :rtype: class `Pandas.DataFrame`
        """
        
        pubs_ = []
        payload = {}
        headers = {'accept': 'application/json, text/plain, */*'}
        pub_df = pd.DataFrame({"user_id": [user_id] * len(pub_ids)})
        url_ = self.univ_details['PROFILE_URL']

        if len(pub_ids) > 0:
            for idx in pub_ids:
                pub_url = "".join([url_, str(idx)])
                response_str = get_request(
                    url=pub_url, headers=headers, data=payload)
                try:
                    dict_ = json.loads(response_str)
                    pubs_.append(dict_)
                except BaseException:
                    pubs_.append({"id": idx})
            return pd.concat([pub_df, pd.DataFrame(pubs_)], axis=1)

        else:
            return pd.DataFrame({"user_id": [user_id]})

    def create_univ_publication_data(self):
        """ Main function which will create the publication data for all the users of a university
        
        :param None: 
        
        :return: None
        """

        # ADD DESCRIPTION
        user_pub_list = [
            (i, j) for i, j in zip(
                self.user_df["User_id"].tolist(), self.user_df["Publications"].tolist())]
        
        #TEST_CODE 
        # user_pub_list = user_pub_list[:100]
        pub_tokens = parallelize(
            self.n_cores,
            func=self.get_publication_ids,
            arg1=user_pub_list)
        user_pub_dict = {k: v for d in pub_tokens for k, v in d.items()}

        # ADD DESCRIPTION
        pub_df = pd.Series(user_pub_dict, name='publication_id').rename_axis(
            'user_id').explode().reset_index()
        pub_list = [(i, j) for i, j in user_pub_dict.items()]

        # ADD DESCRIPTION
        self.pub_data = parallelize(
            self.n_cores,
            func=self.create_user_publication_data,
            arg1=pub_list)

    def save_user_publications(self, univ_name='TAMU'):
        """Function to save the publication details of each user 
        
        :param univ_name: University name of the user - which determines the file names for saving.
        :type univ_name: `str1
                
        """

        pub_final = pd.concat(self.pub_data)
        pub_final = pub_final[['user_id',
                               'id',
                               'class',
                               'title',
                               'abstract',
                               'publicationVenue',
                               'authors',
                               'publicationDate',
                               'publisher',
                               'identifier',
                               'doi',
                               'pageStart',
                               'pageEnd',
                               'volume',
                               'issue',
                               'altmetricScore',
                               'citationCount',
                               'authorList',
                               'type',
                               'modTime',
                               '_links',
                               'bookTitle',
                               'keywords']].reset_index(drop=True)

        pub_final.to_csv(
            os.path.join(
                self.output_path,
                self.publication_file_name),
            index=False,
            escapechar='\\')


if __name__ == "__main__":

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
    
    print("\n\nExtracting Publications")
    
    try:
        params = yaml.safe_load(open(args.config_file))
    except BaseException:
        print(f'Error loading parameter file: {args.config_file}.')
        sys.exit(1)

    publication_data = Extract_Publications(
        n_cores=args.n_cores,
        univ_name=args.univ_name,
        params=params)

    publication_data.create_univ_publication_data()

    publication_data.save_user_publications()
    
    print("TASK COMPLETED : Successfully extracted publications")