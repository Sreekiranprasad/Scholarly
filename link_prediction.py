import os
import sys

import ast
import yaml

import requests
import argparse

import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm

from multiprocessing import Pool

from helpers import *
from automatic_keyword_generator import *


def get_authors(row):
    try:
        list_ = ast.literal_eval(row)
        author = []
        for i in list_:
            author.append(i['label'])
        return ';'.join(author)
    except:
        return ''
    
    
def get_coauthor_weights(author, authors):
    
    co_authors = [i for i in authors if author in i]
    author_counts = {}
    for list_ in co_authors:
        
        sub_list = list_.split(';')
        try:
            sub_list.remove(author)
        except:
            continue
        
        if sub_list:
            for i in sub_list:
                if (author, i) in author_counts:
                    author_counts[(author, i)]+=1
                else:
                    author_counts[(author, i)]=1
    return author_counts

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
        '--n_cores',
        metavar='CPU_COUNT',
        type=int,
        default=0,
        help='No of CPU threads to be used')
    args = parser.parse_args()
    
    # Read configuration file. If not successfull end the program
    try:
        params = yaml.safe_load(open('config.yml'))
    except BaseException:
        print(f'Error loading parameter file: {args.config_file}.')
        sys.exit(1)

    df = pd.read_csv(os.path.join(params['OUTPUT_PATH'],'PublicationDataset.csv'))

    # Extracts authors
    authors = [get_authors(i) for i in df['authors']]

    unique_scholars = []
    for i in authors:
        unique_scholars.extend(i.split(';'))
    unique_scholars = [i for i in set(unique_scholars) if i!='']


    user_pub_list = [(i, authors) for i in unique_scholars]    
    auth_weights = parallelize(
        args.n_cores,
        func=get_coauthor_weights,
        arg1=user_pub_list)

    coauthor_dict = {}
    for i in auth_weights:
        coauthor_dict.update(i)

    G = nx.Graph()
    nodes = set([author for pair in coauthor_dict for author in pair])
    for node in nodes:
        G.add_node(node)

    for edge, weight in coauthor_dict.items():
        G.add_edge(edge[0], edge[1], weight=weight)
        
    # Learn node embeddings
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=20)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    model.save(params['OUTPUT_PATH']+"node2vec.model")
    
    coauthors_dict = {}
    for author in tqdm(unique_scholars):

        coauthors = []
        for other_author in unique_scholars:
            if author != other_author:
                try:
                    similarity = np.dot(model.wv[author], model.wv[other_author])
                    coauthors.append((other_author, similarity))
                except KeyError:
                    continue
        coauthors.sort(key=lambda x: x[1], reverse=True)
        coauthors_dict[author] = [coauthor for coauthor in coauthors[:args.top_k]]


    # Create a list of tuples with the author, coauthor and score
    rows = []
    for author, coauthors in coauthors_dict.items():
        for coauthor, score in coauthors:
            rows.append((author, coauthor, round(score, 2)))

    # Create a Pandas DataFrame from the list of tuples
    recommend_df = pd.DataFrame(rows, columns=["Author", "Coauthor", "Score"])

    # Save the recommendation
    save_pandas_to_csv(
        df=recommend_df,
        output_path=os.path.join(
            params['OUTPUT_PATH'],
            params['COAUTHORSHIP_FILENAME']),
        index=False)