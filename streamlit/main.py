#basics
import pandas as pd
import numpy as np
import joblib
from pickle5 import pickle
from PIL import Image
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

#streamlit
import streamlit as st
import SessionState
from load_css import local_css
local_css("./streamlit/style.css")

DEFAULT = '< PICK A VALUE >'
def selectbox_with_default(text, values, default=DEFAULT, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

#helper functions
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import src.clean_dataset as clean

@st.cache(allow_output_mutation=True)
def neuralqa():
    
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", 
                                              use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", 
                                                          return_dict=False)

    bi_encoder = SentenceTransformer('nq-distilbert-base-v1')
    return tokenizer, model, bi_encoder


sys.path.pop(0)

#%%
#1. load in complete transformed and processed dataset for pre-selection and exploration purpose
df = pd.read_csv('./data/taxonomy_final.csv')
df_columns = df.drop(columns=['PIMS_ID', 'all_text_clean', 'all_text_clean_spacy',  'hyperlink',
 'title',
 'leading_country',
 'grant_amount',
 'country_code',
 'lon',
 'lat'])
    
#2 load corpus embeddings for neural QA:
corpus_embeddings = pickle.load(open("./data/splitted_corpus_embeddings.pkl", 'rb'))

#%%
session = SessionState.get(run_id=0)

#%%
#title start page
st.title('Machine Learning for Nature Climate Energy Portfolio')

sdg = Image.open('./streamlit/logo.png')
st.sidebar.image(sdg, width=200)
st.sidebar.title('Settings')


st.header("Try Neural Question Answering.")
returns = st.sidebar.slider('Maximal number of answer suggestions:', 1, 10, 5)
#        examples=["", 
#                  "what's the problem with the artibonite river basin?", 
#                  "what are threads for the machinga and mangochi districts of malawi?",
#                  "how can we deal with rogue swells?"]

#example = st.selectbox('Examples:', [k for k in examples], format_func=lambda x: 'Select an Example' if x == '' else x)
        
question = st.text_input('Type in your question (be as specific as possible):')

#load and split dataframe:
wrapped, splitted = clean.split_at_length(df, 'all_text_clean', 512)
passages = splitted.text.tolist()
passage_id = splitted.PIMS_ID.tolist()

#if st.button('Evaluate'):
if question != "":
    with st.spinner('Processing all logframes and finding best answers...'):
        tokenizer, model, bi_encoder = neuralqa()
        top_k = returns  # Number of passages we want to retrieve with the bi-encoder
        question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
        
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  
        
        #define lists
        matches = []
        ids = []
        scores = []
        answers = []

        for hit in hits:
            matches.append(passages[hit['corpus_id']])
            ids.append(passage_id[hit['corpus_id']])
            scores.append(hit['score'])
            
        for match in matches:
            inputs = tokenizer.encode_plus(question, match, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = model(**inputs)

            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            
            answers.append(answer)
        
            
        # generate result df
        df_results = pd.DataFrame(
            {'PIMS_ID': ids,
                'answer': answers,
                'context': matches,
                "scores": scores
            })

        
        
        st.header("Retrieved Answers:")
        for index, row in df_results.iterrows():
            green = "<span class='highlight turquoise'>"+row['answer']+"<span class='bold'>Answer</span></span>"
            row['context'] = row['context'].replace(row['answer'], green)
            row['context'] = "<div>"+row['context']+"</div>"
            st.markdown(row['context'], unsafe_allow_html=True)
            st.write("")
            st.write("Relevance:", round(row['scores'],2), "PIMS_ID:", row['PIMS_ID'])
            st.write("____________________________________________________________________")
            
        df_results.set_index('PIMS_ID', inplace=True)
        st.header("Summary:")
        st.table(df_results)
                
                        
#%%
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
st.write('           ')
if st.button("Run again!"):
  session.run_id += 1

#%%
from pathlib import Path
p = Path('.')
