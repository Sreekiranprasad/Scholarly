import os
import sys

import json
import yaml
import argparse
import requests

import pandas as pd
import numpy as np

from tqdm import tqdm

from helpers import extract_json, save_pandas_to_csv

import pdb


def get_userid(user_dict, key):
    """ Function to get User IDs from JSON 

        :param user_dict: The URL from which response is to be retrieved
        :type user_dict: `JSON`
        
        :return: User ID
        :rtype: `str`
    """
    try:
        idx = user_dict[key]
    except BaseException:
        idx = np.nan
    return idx


class extract_user_profiles():
    """ Class which can extract profiles of all users from a university
    """

    def __init__(self, univ_name, output_path):
        """ Constructor

        :param univ_name: Name of the univeristy
        :type univ_name: `str`
        
        :return: None
        """

        self.output_path = params['OUTPUT_PATH'] if output_path == '' else output_path
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.output_path )
        self.profile_url = params['UNIV_DETAILS'][args.univ_name]['PROFILE_URL']
        self.base_url = params['UNIV_DETAILS'][args.univ_name]['BASE_URL']
        self.end_url = params['UNIV_DETAILS'][args.univ_name]['END_URL']
        self.sub_json = extract_json(self.base_url, self.end_url, 1)
        self.n_scholars = self.sub_json['page']['totalElements']
        self.scholars_dataset = params['SCHOLARS_DATASET']
        print("Total Scholars: ", self.n_scholars)

        # path = os.path.join(os.getcwd(), "Test_Folder")
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def extract_info(self, url, user_id):
        """ Function to extract a particular user's information from general university URL  

        :param url: The URL from which response is to be retrieved
        :type url: `str`
        :param user_id: ID of the particular scholar
        :type user_id: `str`
        
        :return: None
        """

        self.user_url = url + user_id
        payload = {}
        headers = {
            'accept': 'application/json, text/plain, */*'
        }

        response = requests.request(
            "GET", self.user_url, headers=headers, data=payload)
        response_str = response.text
        self.user_dict = json.loads(response_str)

        return None

    def get_name(self):
        """ Function to extract name of the Scholar from University Page 

        :param None: 
        
        :return: Name of the Scholar
        :rtype: `str`
        """

        return self.user_dict['name']

    def get_email(self):
        """ Function to extract email of the Scholar from University Page 

        :param None: 
        
        :return: Email of the Scholar
        :rtype: `str`
        """
        try:
            return self.user_dict['primaryEmail']
        except BaseException:
            return None

    def get_title(self):
        """ Function to extract Prefered title of the Scholar from University Page 

        :param None: 
        
        :return: Preferred title of the Scholar
        :rtype: `str`
        """
        return self.user_dict['preferredTitle']

    def get_department_info(self):
        """ Function to extract Department info (including course area) of the Scholar from University Page 

        :param None: 
        
        :return: Department info  of the Scholar
        :rtype: `str`
        """

        try:
            position_list = self.user_dict["positions"]

            course_area = []
            department = []
            try:
                for p in position_list:
                    course_area.append(p['organizations'][0]['label'])
                    department.append(
                        p['organizations'][0]['parent'][0]['label'])
            except BaseException:
                pass
            return "||".join(course_area), "||".join(department)
        except BaseException:
            return None, None

    def get_overview(self):
        """ Function to extract Overview of the Scholar from University Page 

        :param None: 
        
        :return: Overview  of the Scholar
        :rtype: `str`
        """

        try:
            return self.user_dict['overview']
        except BaseException:
            return None

    def get_keywords(self):
        """ Function to extract keywords of the Scholar from University Page 

        :param None: 
        
        :return: Keywords  of the Scholar
        :rtype: `str`
        """
        try:
            return "||".join(self.user_dict['keywords'])
        except BaseException:
            return None

    def get_npublications(self):
        """ Function to get the no of publciations of the Scholar from University Page 

        :param None: 
        
        :return: Publications  of the Scholar
        :rtype: `str`
        """
        
        try:
            return len(self.user_dict["publications"])
        except BaseException:
            return None

    def get_publications(self):
        """ Function to extract publications of the Scholar from University Page 

        :param None: 
        
        :return: Publications  of the Scholar
        :rtype: `str`
        """
        try:
            return json.dumps(self.user_dict['publications'])
        except BaseException:
            return None

    def get_research(self):
        """ Function to Research areas of the Scholar from University Page 

        :param None: 
        
        :return: Research areas  of the Scholar, Length of research_areas
        :rtype: `Tuple (List, Int)`
        """

        try:
            r_list = []
            for d in self.user_dict["researcherOn"]:
                r_list.append(d['label'])
            return "||".join(r_list), len(r_list)
        except BaseException:
            return None, 0

    def get_awards(self):
        """ Function to Research areas of the Scholar from University Page 

        :param None: 
        
        :return: Research areas  of the Scholar, Length of research_areas
        :rtype: `Tuple (List, Int)`
        """
        
        try:
            a_list = []
            for d in self.user_dict["awardsAndHonors"]:
                a_list.append(d['label'])
            return "||".join(a_list), len(a_list)
        except BaseException:
            return None, 0

    def get_organizations(self):
        """ Function to extract Organizations of the Scholar from University Page 

        :param None: 
        
        :return: Organizations  of the Scholar
        :rtype: `str`
        """
        try:
            return self.user_dict['organizations']
        except BaseException:
            return None

    def get_department(self):
        """ Function to extract Department of the Scholar from University Page 

        :param None: 
        
        :return: Department of the Scholar
        :rtype: `str`
        """
        try:
            return self.user_dict['schools']
        except BaseException:
            return None

    def get_netid(self):
        """ Function to extract NetID (University Unique Identifier) of the Scholar from University Page 

        :param None: 
        
        :return: NetID  of the Scholar
        :rtype: `str`
        """
        try:
            return self.user_dict['netid']
        except BaseException:
            return None

    def get_profile(self, url, user_id):
        """ Function to extract all details of a scholar from University Page 

        :param url: The base university URL from which Scholars' data can be extracted by appending their user_ids
        :type url: `str`
        :param user_id: The university provided User ID of the scholar
        :type user_id: `str`
        
        :return: Scholar Data in the form of Pandas.DataFrame
        :rtype: `Pandas.DataFrame`
        """
        
        self.extract_info(url, user_id)

        course, dept = self.get_department_info()
        research, r_len = self.get_research()
        awards, a_len = self.get_awards()

        try:
            org = "||".join(self.get_organizations())
        except BaseException:
            org = None
        try:
            user_data = pd.DataFrame({

                "User_id": user_id,
                "Netid": self.get_netid(),
                "Name": self.get_name(),
                "Email": self.get_email(),
                "Type": self.get_title(),
                "Overview": self.get_overview(),
                "Keywords": self.get_keywords(),
                "n_publications": self.get_npublications(),
                "Publications": self.get_publications(),
                "Research": research,
                "n_research": r_len,
                "Awards": awards,
                "n_awards": a_len,
                "Organizations": org,
                "Course": course,
                "Department": dept
            }, index=[0])
        except BaseException:
            user_data = pd.DataFrame(
                {
                    'columns': [
                        "User_id",
                        "Netid",
                        "Name",
                        "Email",
                        "Type",
                        "Overview",
                        "Keywords",
                        "n_publications",
                        "Research",
                        "n_research",
                        "Awards",
                        "n_awards",
                        "Organizations",
                        "Course",
                        "Department"]})
        return user_data

    def extract_profiles(self):
        """ Function to compile Scholar data of a particular university.
        The function will first identify the total number of scholars in a university and then get basic summary available for each scholar.

        :param None: 
        
        :return: None
        """

        # From the main URL, identify the number of scholars
        final_json = extract_json(self.base_url, self.end_url, self.n_scholars)
        user_jsons = final_json['_embedded']['individual']

        user_ids = [get_userid(d, "id") for d in user_jsons]
        user_ids = [i for i in user_ids if i is not np.nan]

        print("Total ids extracted for scholars", len(user_ids))

        url = self.profile_url + user_ids[1]
        payload = {}
        headers = {
            'accept': 'application/json, text/plain, */*'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        response_str = response.text
        user_dict = json.loads(response_str)

        # For each scholar, go to his/her summary page and extract relevant data
        user_list = []

        # #TODELETE
        # user_ids = user_ids[:10]
        for idx in tqdm(user_ids):
            user_list.append(self.get_profile(self.profile_url, idx))

        # Save User profiles
        df = pd.concat(user_list).reset_index(drop=True)

        save_pandas_to_csv(
            df=df,
            output_path=os.path.join(
                self.output_path,
                self.scholars_dataset),
            index=False)


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
        '--output_path',
        metavar='OUTPUT_PATH',
        type=str,
        default='',
        help='Path for saving output file')
    args = parser.parse_args()
    print("\n\nCreating User Profile")
    try:
        params = yaml.safe_load(open(args.config_file))
    except BaseException:
        print(f'Error loading parameter file: {args.config_file}.')
        sys.exit(1)

    profile_extractor_object = extract_user_profiles(
        args.univ_name, args.output_path)
    profile_extractor_object.extract_profiles()
    
    print("TASK COMPLETED : Successfully created User profiles")