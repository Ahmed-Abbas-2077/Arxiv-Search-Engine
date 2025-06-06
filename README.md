# Arxiv-Search-Engine

Information retrieval (IR) is the art and craft of identifying and presenting relevant and informative data, mainly in the form of textual representations.
ArXiv is an open-access repository of electronic preprints and postprints approved for posting after moderation, but not peer reviewed. It consists
of scientific papers in many fields such as computer science, physics, and biology. It is indeed a common predicament of researchers that they
often find it particularly difficult to collect relevant and timely data, be it papers, articles, or manuscripts to support their literature reviews. As such,
we have endeavored to leverage modern IR techniques to enhance the productivity of researchers and augment the capabilities thereof. We have
made use of several IR and natural language processing (NLP) libraries including PyTerrier, NLTK, and Gensim, in addition to custom functions of
our own design. Our proprietary search engine, ArXiver, is evaluated against a host of queries and with controlled parameterization and diverse
inputs. We included query expansion with pseudo-relevance feedback (PRF) and Word2Vec embeddings, so as to enhance the relevance and
recall of search results. For ranking and relevance scoring, we used term frequency-inverse document frequency (TF-IDF) to weight terms based
on their rarity in the given corpus.


## Features
- Search for papers by title, abstract, or authors
- Query expansion with pseudo-relevance feedback (PRF)
- Word2Vec embeddings for better relevance
- Term frequency-inverse document frequency (TF-IDF) for ranking
- Custom functions for data preprocessing and search
- Evaluation against a host of queries


## Installation
First, install the requirements:
```bash
pip install -r requirements.txt
```

Then, download the arxiv dataset from https://www.kaggle.com/datasets/Cornell-University/arxiv

## Usage
Run the notebook `search.ipynb` to see how the search engine works interactively. You can also use the `search.py` script to search from the command line:

```bash
# Interactive mode - prompts for query and search method
python search.py

# Direct search with a query
python search.py "quantum computing"

# Specify search method (simple, prf, w2v, or combined)
python search.py --method prf "machine learning"

# Display more results
python search.py --results 10 "neural networks"
```

The script offers several search methods:
- `simple`: Basic TF-IDF ranked retrieval
- `prf`: Query expansion using Pseudo-Relevance Feedback
- `w2v`: Query expansion using Word2Vec embeddings
- `combined`: Both PRF and Word2Vec expansion (default)


# Contributing
We welcome contributions! Please fork the repository and submit a pull request with your changes. Make sure to follow the coding style and include tests for new features.