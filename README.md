# Neural Question Answering:
Question answering (QA) is a computer science discipline within the fields of information retrieval and natural language processing (NLP), which is concerned with building systems that automatically answer questions posed by humans in a natural language.
A question answering implementation, usually a computer program, may construct its answers by querying a structured database of knowledge or information, usually a knowledge base. More commonly, question answering systems can pull answers from an unstructured collection of natural language documents.
[Source](https://en.wikipedia.org/wiki/Question_answering).


Try this application to ask open questions to the UNDP project documents using a Neural QA pipeline powered by sentence transformers to build corpus embeddings and ranking of paragraphs. For retrieval a pre-trained transformer model for extractive QA is applied. The results are highlighted in html.

[![QA](https://github.com/jonas-nothnagel/ClosedDomainQA/blob/master/img/neural_qa.gif)](#features)

Try the App: [Streamlit](https://share.streamlit.io/jonas-nothnagel/closeddomainqa/streamlit/main.py)!
# Setup

Please use Python <= 3.7 to ensure working pickle protocol.

Clone the repo to your local machine:
```
https://github.com/jonas-nothnagel/ClosedDomainQA.git
```
To only run the web application install the dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```
On the first run, the app will download several transformer models (3-4GB) and will store them on your local system. To start the application, navigate to the streamlit folder and simply run:

To start the application, navigate to the streamlit folder and simply run:
```
streamlit run main.py
```
# Upcoming
* Telegram Chatbot Implementation.
* Refined Ranking.
* Refined Extractive QA
