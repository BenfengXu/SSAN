import os
import json
import logging

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from dataset import DocREDProcessor
from dataset import docred_convert_examples_to_features as convert_examples_to_features
from run_docred import setup_SSAN_parser


def file_to_examples_and_features(args):
    """Convert an SSAN input file into DocREDExample and DocREDInputFeatures objects."""
    processor = DocREDProcessor()
    label_map = processor.get_label_map(args.data_dir)
    num_labels = len(label_map.keys())


    logging.info("Loading examples from file")
    examples = processor.get_examples_from_file(args.input_file)


    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=None
    )

    logging.info("Converting examples to features")
    features = convert_examples_to_features(
        examples,
        args.model_type,
        tokenizer,
        max_length=args.max_seq_length,
        max_ent_cnt=args.max_ent_cnt,
        label_map=label_map
    )

    logging.info("Finished converting examples to features")

    return examples, features


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = setup_SSAN_parser()
    parser.add_argument("--input_file", required=True, help="Input file for preprocessing.")
    args = parser.parse_args()


    examples, features = file_to_examples_and_features(args)


    input_dir, input_fname = os.path.split(args.input_file)
    input_dir = os.path.abspath(input_dir)

    # Write example and feature files next to input file
    write_examples_to = os.path.join(input_dir, f"{os.path.splitext(input_fname)[0]}_examples.json")
    write_features_to = os.path.join(input_dir, f"{os.path.splitext(input_fname)[0]}_features.json")


    logging.info("Json serializing examples and features...")
    json_examples = [e.to_json_string() for e in examples]
    json_features = [f.to_json_string() for f in features]

    logging.info(f"Writing examples to {write_examples_to}")
    with open(write_examples_to, "w") as f:
        json.dump(json_examples, f)

    logging.info(f"Writing features to {write_features_to}")
    with open(write_features_to, "w") as f:
        json.dump(json_features, f)


if __name__ == "__main__":
    main()
