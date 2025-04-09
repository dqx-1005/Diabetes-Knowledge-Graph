#  Diabetes Knowledge Graph

This is a TensorFlow implementation of DiabetesKG, and the code includes the following modules:

* Source (Diabetes datasets from https://tianchi.aliyun.com/competition/entrance/231687)

* Named Entity Recognition for diabetes-related concepts

* Extracted entity vectors are saved and used for knowledge graph construction in Neo4j


## Main Requirements

* tensorflow 1.12
* python 3.6


## Description

* main.py  
  * Main script that handles the model training and evaluation process.
* predict_now.py
  * Find mappings in knowledge graph
* loader.py
  * Preprocess sentences in diabetes dataset



## Running the code

1. Install the required dependency packages
2. To reproduce the results, please use the command `python main.py`