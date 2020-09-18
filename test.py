#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:33:37 2020

@author: jonas

@title: haystack
"""


import logging
import subprocess
import time

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
