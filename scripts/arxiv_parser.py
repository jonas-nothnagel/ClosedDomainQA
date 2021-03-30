# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:03:56 2021

@author: jonas.nothnagel

@title: arxiv scraper test
"""
import arxiv
import pandas as pd
import pickle
#%%
def query_arxiv(string_query, results):
    
    abstracts = []
    title =[]
    authors =[]
    url = []
    paper = arxiv.query(query=string_query, max_results=results)
    
    for p in paper:
        for key, value in p.items():
            if key == "summary":
                abstracts.append(value)
            if key == "title":
                title.append(value)
            if key == "authors":
                authors.append(value)
            if key == "arxiv_url":
                url.append(value)    
                
    df = pd.DataFrame(
    {'title': title,
     'summary': abstracts,
     'authors': authors,
     "url": url
    })
    
    df.to_csv('arxiv_results_15000.csv')
    with open('feedparser_results.pickle', 'wb') as handle:
        pickle.dump(paper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return paper, df
#%%

paper, df = query_arxiv("abs:disaster risk management", 100000000)

