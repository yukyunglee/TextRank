import argparse
import os
import time
from typing import Optional, Any

import spacy

from utils import *
from keyword_extractor import *
from sentence_extractor import *


def main(args):

    input_data: Optional[Any] = read_dataset(args.dataset_directory)

    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)

    # load spacy tokenizer
    # sm == 'small' , lg == 'large'
    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_lg")

    if args.mode == "keyword":
        start = time.time()

        results = key_word_summary(
            nlp, input_data, args.min_sim, args.window_size, args.keyword_top_k
        )
        save_path = os.path.join(args.save_directory, "keyword_sum_result.pkl")
        save_result(save_path, results)

        print("Keyword sum : save_results done")
        print("processing time : ", time.time() - start)

    elif args.mode == "sentence":
        start = time.time()

        results = sentence_summary(nlp, input_data, args.min_sim, args.sentence_top_k)
        save_path = os.path.join(args.save_directory, "sentence_sum_result.pkl")
        save_result(save_path, results)

        print("Sentence sum : save_results done")
        print("processing time : ", time.time() - start)


if __name__ == "__main__":

    ###########################################
    # python -m spacy download en_core_web_sm #
    # python -m spacy download en_core_web_lg #
    ###########################################

    parser = argparse.ArgumentParser()

    # setting
    parser.add_argument("--dataset_directory", default="./dataset/ext_val.pickle")
    parser.add_argument("--save_directory", default="./results")
    parser.add_argument("--mode", default="keyword", help="sentence")

    parser.add_argument("--min_sim", default=0.3, help="minimum similarity")
    parser.add_argument("--window_size", default=3)
    parser.add_argument("--keyword_top_k", default=15)
    parser.add_argument("--sentence_top_k", default=5)

    main_args = parser.parse_args()
    main(main_args)
