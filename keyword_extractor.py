import itertools
import networkx as nx
from numpy import dot
from numpy.linalg import norm
import statistics


def get_unique_vocab(key_word, stop_words=None):

    # stopword change set

    unique_vocab = set()
    add_vocab = unique_vocab.add

    for element in [x for x in key_word if x not in unique_vocab]:
        add_vocab(element)

    if stop_words is not None:
        unique_vocab.difference(stop_words)

    return unique_vocab


def make_word_graph(vocabs, key_word, min_sim):
    word_graph = nx.Graph()  # initialize an undirected graph
    word_graph.add_nodes_from(vocabs)

    nodePairs = list(itertools.combinations(vocabs, 2))

    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        node1 = pair[0]
        node2 = pair[1]

        cos_sim = dot(key_word[pair[0]], key_word[pair[1]]) / (
            norm(key_word[pair[0]]) * norm(key_word[pair[1]])
        )

        if cos_sim > min_sim:
            word_graph.add_edge(node1, node2, weight=cos_sim)

    return word_graph


def extract_words(word_graph, vocabs, key_word, window_size, top_k):

    word_set_list = list(vocabs)
    textlist = list(key_word.keys())

    calculated_page_rank = nx.pagerank(
        word_graph, alpha=0.85, max_iter=100, weight="weight"
    )
    # alpha means damping factor

    keywords = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    one_third = len(word_set_list) // window_size
    keywords = keywords[0 : one_third + 1]

    modified_keywords = set([])

    dealt_with = set([])
    i = 0
    j = 1
    while j < len(textlist):
        first = textlist[i]
        second = textlist[j]
        if first in keywords and second in keywords:
            keywords = first + " " + second
            modified_keywords.add(keywords)
            dealt_with.add(first)
            dealt_with.add(second)
        else:
            if first in keywords and first not in dealt_with:
                modified_keywords.add(first)

            if (
                j == len(textlist) - 1
                and second in keywords
                and second not in dealt_with
            ):
                modified_keywords.add(second)

        i = i + 1
        j = j + 1

    new_extract = {}
    for vocab in list(modified_keywords):
        if vocab not in calculated_page_rank:
            new = vocab.split()
            new_score = [calculated_page_rank[n] for n in new]
            new_extract[vocab] = statistics.mean(new_score)
        else:
            new_extract[vocab] = calculated_page_rank[vocab]

    words = sorted(new_extract, key=new_extract.get, reverse=True)[
        : -len(new_extract) + top_k
    ]

    return words


# main_function
def key_word_summary(nlp, dataset, min_sim, wondow_size, top_k):

    key_sum_result = []

    for idx, cur_data in enumerate(dataset):

        # cur_data == one article
        # in textrank input should be string

        # change src to string
        article = [" ".join(src["src"]) for src in cur_data]
        key_word = {}
        tag_list = ["NN", "JJ", "NNP"]

        for sent in article:
            results = nlp(sent)
            for result in results:
                if (result.tag_ in tag_list) and (result.text not in key_word):
                    key_word[result.text] = result.vector

        vocabs = get_unique_vocab(key_word)
        word_graph = make_word_graph(vocabs, key_word, min_sim=min_sim)
        extracted_key_word = extract_words(
            word_graph, vocabs, key_word, window_size=wondow_size, top_k=top_k
        )
        key_sum_result.append(extracted_key_word)

    return key_sum_result
