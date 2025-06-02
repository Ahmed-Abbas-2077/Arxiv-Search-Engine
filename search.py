#!/usr/bin/env python
# filepath: e:\Zewail\Sophomore Year\DSAI 201\Project\search.py

"""
ArXiv Search Engine Script

This script implements a search engine for ArXiv papers using:
- Text preprocessing (tokenization, lemmatization, stopword removal)
- PyTerrier for indexing and retrieval
- TF-IDF scoring
- Query expansion with PRF and Word2Vec
"""

import os
import pandas as pd
import numpy as np
import pyterrier as pt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import argparse
from tqdm import tqdm
import math

# Initialize PyTerrier
pt.java.add_package('com.github.terrierteam', 'terrier-prf', '-SNAPSHOT')
pt.java.init()  # forces java initialization

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "arxiv-uno.csv")
INDEX_PATH = os.path.join(BASE_DIR, "arxIndexer")


class ArxivSearchEngine:
    def __init__(self, data_path=DATA_PATH, index_path=INDEX_PATH):
        """Initialize the search engine with data and index paths."""
        self.data_path = data_path
        self.index_path = index_path
        self.stop_words = set(stopwords.words('english'))
        self.wnl = WordNetLemmatizer()
        self.wordnet = nltk.corpus.wordnet

        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)

        # Load or create index
        if os.path.exists(self.index_path) and os.path.isdir(self.index_path):
            print(f"Loading index from {self.index_path}...")
            self.index = pt.IndexFactory.of(self.index_path)
        else:
            print("Index not found. Creating new index...")
            self.create_index()

        # Load or train Word2Vec model
        self.word_embeddings = self._train_word2vec()

    def get_pos(self, nltk_tag):
        """Convert NLTK POS tag to WordNet POS tag."""
        if nltk_tag.startswith('J'):
            return self.wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return self.wordnet.VERB
        elif nltk_tag.startswith('N'):
            return self.wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return self.wordnet.ADV
        else:
            return self.wordnet.NOUN

    def lemmatize(self, tokens):
        """Lemmatize tokens using their part of speech."""
        tagged = nltk.pos_tag(tokens)  # list of tuples (word, tag)
        lemmatized = [
            self.wnl.lemmatize(word, self.get_pos(tag))
            for word, tag in tagged
        ]
        return lemmatized

    def preprocess(self, text):
        """Preprocess text: tokenize, lowercase, remove stopwords, lemmatize."""
        if pd.isna(text):
            return ""
        tokens = word_tokenize(text)  # tokenize
        tokens = [word.lower() for word in tokens]  # lowercase
        # remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = self.lemmatize(tokens)  # lemmatize
        return ' '.join(tokens)  # back to string

    def create_index(self):
        """Create a PyTerrier index from the dataframe."""
        # Prepare documents for indexing
        docs = []
        for i, row in self.df.iterrows():
            title = row['title'] if pd.notnull(row['title']) else ""
            authors = row['authors'] if pd.notnull(row['authors']) else ""
            abstract = row['abstract'] if pd.notnull(row['abstract']) else ""
            docs.append({
                "docno": str(row['docno']),
                "text": f"Title: {title} | Authors: {authors} | Abstract: {abstract}"
            })

        # Create and save the index
        indexer = pt.terrier.IterDictIndexer(
            index_path=self.index_path, overwrite=True)
        index_ref = indexer.index(docs)
        self.index = pt.IndexFactory.of(index_ref)
        print(f"Index created at {self.index_path}")

    def _train_word2vec(self):
        """Train or load a Word2Vec model."""
        model_path = os.path.join(os.path.dirname(
            self.index_path), "word2vec.model")

        try:
            if os.path.exists(model_path):
                print(f"Loading Word2Vec model from {model_path}...")
                return Word2Vec.load(model_path).wv
            else:
                print(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Will train a new model instead.")

        print("Training Word2Vec model...")
        # Tokenize abstracts for training
        self.df['abstract_tokens'] = self.df['abstract'].apply(
            lambda x: word_tokenize(x) if pd.notnull(x) else [])

        # Train Word2Vec model
        model = Word2Vec(sentences=self.df["abstract_tokens"],
                         vector_size=100,
                         window=5,
                         min_count=5,
                         sg=1,
                         workers=4,
                         epochs=30)

        # Save the model
        model.save(model_path)
        print(f"Word2Vec model saved to {model_path}")

        return model.wv

    def search_term(self, term):
        """Search for a specific term in the index."""
        term = self.preprocess(term)
        results = []

        try:
            pointer = self.index.getLexicon()[term]
            for posting in self.index.getInvertedIndex().getPostings(pointer):
                results.append({
                    "doc_id": posting.getId(),
                    "length": posting.getDocumentLength(),
                    "frequency": posting.getFrequency()
                })
            return results
        except KeyError:
            print(f"'{term}' not found in the index.")
            return []

    def ranked_retrieval(self, query):
        """Simple ranked retrieval using TF-IDF."""
        query = self.preprocess(query)
        tf_idf = pt.terrier.Retriever(self.index, wmodel="TF_IDF")
        results = tf_idf.search(query)  # This line was missing
        return results

    def expand_query_with_word2vec(self, query, topn=3):
        """Expand query using Word2Vec embeddings."""
        expanded_terms = []
        query = self.preprocess(query)

        for term in query.split():
            if term in self.word_embeddings:
                sim = self.word_embeddings.most_similar(term, topn=topn)
                sim_terms = [x[0] for x in sim]
                expanded_terms.extend(sim_terms)

        expanded_query = query + " " + " ".join(expanded_terms)
        return expanded_query

    def prf_search(self, query, top_docs=10, expansion_terms=5):
        """Perform pseudo-relevance feedback search."""
        query = self.preprocess(query)
        tf_idf = pt.terrier.Retriever(self.index, wmodel="TF_IDF")

        # Use RM3 for query expansion
        rm3_expand = pt.rewrite.RM3(
            self.index, fb_docs=top_docs, fb_terms=expansion_terms)
        prf_pipeline = tf_idf >> rm3_expand >> tf_idf

        results = prf_pipeline.search(query)
        return results

    def combined_search(self, query, w2v_terms=3, prf_docs=10, prf_terms=5):
        """Combine Word2Vec and PRF for query expansion."""
        print(f"Expanding query: '{query}'")
        expanded_query = self.expand_query_with_word2vec(query, topn=w2v_terms)
        print(f"Expanded query: '{expanded_query}'")
        print("Performing PRF search...")
        results = self.prf_search(
            expanded_query, top_docs=prf_docs, expansion_terms=prf_terms)
        print(f"Found {len(results)} results")
        return results

    def display_results(self, results, n=10):
        """Display search results with paper details."""
        if len(results) == 0:
            print("No results found.")
            return

        matched_docs = 0
        for i, (_, row) in enumerate(results.head(n).iterrows()):
            doc_id = row["docno"]
            score = row["score"]

            print(f"\nResult {i+1}:")
            print(f"Document ID: {doc_id}, Score: {score:.4f}")

            # Try to find a match in the dataframe
            try:
                # First try exact match
                paper = self.df[self.df['docno'] == doc_id]

                # If no match, try matching by prefix (e.g., 704.xxxx)
                if len(paper) == 0:
                    prefix = doc_id.split('.')[0]
                    paper = self.df[self.df['docno'].astype(
                        str).str.startswith(prefix)]
                    if len(paper) > 0:
                        # Take the first matching paper
                        paper = paper.iloc[[0]]

                if len(paper) > 0:
                    matched_docs += 1
                    title = paper['title'].values[0] if 'title' in paper.columns else "No title"
                    authors = paper['authors'].values[0] if 'authors' in paper.columns else "Unknown"
                    abstract = paper['abstract'].values[0] if 'abstract' in paper.columns else "No abstract"
                    print(f"Title: {title}")
                    print(f"Authors: {authors}")
                    # Show first 300 chars
                    print(f"Abstract: {abstract[:300]}...")
                else:
                    print("Paper details not found in database.")

            except Exception as e:
                print(f"Error displaying result: {e}")

        print(
            f"\nDisplayed {matched_docs} matched documents out of {min(n, len(results))} results")


def main():
    """Main function to run the search engine from command line."""
    parser = argparse.ArgumentParser(description='ArXiv Search Engine')
    parser.add_argument('query', nargs='*', help='Search query')
    parser.add_argument('--method', choices=['simple', 'prf', 'w2v', 'combined'],
                        default='combined', help='Search method')
    parser.add_argument('--results', type=int, default=5,
                        help='Number of results to display')
    args = parser.parse_args()

    search_engine = ArxivSearchEngine()

    if not args.query:
        # Interactive mode
        while True:
            query = input("\nEnter search query (or 'q' to quit): ")
            if query.lower() == 'q':
                break

            method = input(
                "Search method (simple/prf/w2v/combined) [combined]: ") or 'combined'

            if method == 'simple':
                results = search_engine.ranked_retrieval(query)
            elif method == 'prf':
                results = search_engine.prf_search(query)
            elif method == 'w2v':
                expanded = search_engine.expand_query_with_word2vec(query)
                print(f"Expanded query: {expanded}")
                results = search_engine.ranked_retrieval(expanded)
            else:  # combined
                results = search_engine.combined_search(query)

            search_engine.display_results(results, n=args.results)
    else:
        # Direct query from command line
        query = ' '.join(args.query)
        print(f"Searching for: '{query}' using {args.method} method")

        try:
            if args.method == 'simple':
                results = search_engine.ranked_retrieval(query)
                print(f"Retrieved {len(results)} results")
            elif args.method == 'prf':
                results = search_engine.prf_search(query)
            elif args.method == 'w2v':
                expanded = search_engine.expand_query_with_word2vec(query)
                print(f"Expanded query: {expanded}")
                results = search_engine.ranked_retrieval(expanded)
            else:  # combined
                results = search_engine.combined_search(query)

            if len(results) == 0:
                print("No results found.")
            else:
                print(
                    f"Displaying top {args.results} of {len(results)} results:")
                search_engine.display_results(results, n=args.results)
        except Exception as e:
            print(f"Error during search: {e}")


if __name__ == "__main__":
    main()
