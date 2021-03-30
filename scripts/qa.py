#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:27:19 2021

@author: jonas

@tile: qa_test_haystack
"""
#%%
#import packages
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.preprocessor.preprocessor import PreProcessor

from haystack.document_store.faiss import FAISSDocumentStore

import pandas as pd
import pickle

from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import src.clean_dataset as clean
#%%

#import data
df = pd.read_csv('../data/raw/arxiv_results.csv')
raw_data = pickle.load(open("../data/raw/feedparser_results.pickle", 'rb'))

#%%
#feedparser to df

abstracts = []
title =[]
authors =[]
url = []
published = []
for p in raw_data:
    for key, value in p.items():
        if key == "summary":
            abstracts.append(value)
        if key == "title":
            title.append(value)
        if key == "authors":
            authors.append(value)
        if key == "arxiv_url":
            url.append(value)    
        if key == "published":
            published.append(value)            
df = pd.DataFrame(
{'title': title,
'summary': abstracts,
'authors': authors,
"url": url,
"published": published
})

# remove linebreaks
df['summary'] = df['summary'].astype(str).apply(clean.remove_linebreak)
df['title'] = df['title'].astype(str).apply(clean.remove_linebreak)

# Dataframe to dict for haystack
all_dicts = df[['title', 'summary']].rename(columns={'title':'name','summary':'text'}).to_dict(orient='records')
# %%
# clean data
# preprocessing from haystack
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
    split_overlap=10
)
nested_docs = [preprocessor.process(d) for d in all_dicts]
docs = [d for x in nested_docs for d in x]
# %%
# start FAISS document store and store docs
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

document_store.write_documents(docs)

# %%
# initialise storage
from haystack.retriever.dense import DensePassageRetriever
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=False,
                                  embed_title=True,
                                  use_fast_tokenizers=True)
# %%
