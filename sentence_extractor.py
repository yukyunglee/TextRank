import re
import itertools
import networkx as nx
import numpy as np
from numpy import dot
from numpy.linalg import norm


def make_sentence_graph(sentence, min_sim):
    sentence_graph = nx.Graph()  # initialize an undirected graph
    sentence_graph.add_nodes_from(sentence)

    nodePairs = list(itertools.combinations(sentence, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        node1 = pair[0]
        node2 = pair[1]

        cos_sim = dot(sentence[pair[0]][1], sentence[pair[1]][1]) / (
            norm(sentence[pair[0]][1]) * norm(sentence[pair[1]][1])
        )

        if cos_sim > min_sim:
            sentence_graph.add_edge(node1, node2, weight=cos_sim)

    return sentence_graph


def extract_sentence(sentence_graph, sentence, top_k):
    calculated_page_rank = nx.pagerank(
        sentence_graph, alpha=0.85, max_iter=100, weight="weight"
    )

    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    modified_sentence = sentences[: -len(sentences) + top_k]
    result_sentence = [(sentence[sent][0], sent) for sent in modified_sentence]

    return result_sentence


def sentence_summary(nlp, dataset, min_sim, top_k):
    """

    :type dataset: object
    """
    sentence_sum_result = []

    for idx, cur_data in enumerate(dataset):

        # cur_data == one article
        # in textrank input should be string

        # change src to string
        article = [" ".join(src["src"]) for src in cur_data]
        sentence = {}
        # article 자체로 그래프 생성해야함 - 문장단

        for idx, sent in enumerate(article):
            re_sent = re.sub(r"[^\.\?\!\w\d\s]", "", sent)
            results = nlp(re_sent)

            sentence_vector = [result.vector for result in results]
            sentence[sent] = [idx, np.mean(sentence_vector, axis=0)]

        sentence_graph = make_sentence_graph(sentence, min_sim=min_sim)

        extracted_sentence = extract_sentence(sentence_graph, sentence, top_k=top_k)
        sentence_sum_result.append(extracted_sentence)

    return sentence_sum_result
