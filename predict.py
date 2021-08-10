import os
import json
import logging

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from dataset import DocREDProcessor, DocREDExample, DocREDInputFeatures
from model import BertForDocRED, RobertaForDocRED
from run_docred import set_seed, setup_SSAN_parser, predict_from_examples_and_features


def deserialize_examples_file(path_to_examples):
    """Deserialize a file of serialized DocREDExample objects."""
    with open(path_to_examples, "r") as f:
        example_strings = json.load(f)

    return (DocREDExample(**json.loads(s)) for s in example_strings)

def deserialize_features_file(path_to_features):
    """Deserialize a file of serialized DocREDInputFeatures objects."""
    ndarray_fields = ["ent_mask", "ent_distance", "structure_mask", "label", "label_mask"]
    decode_numpy_fields = lambda feature_dict: {k, v}

    with open(path_to_features, "r") as f:
        features_strings = json.load(f)

    for s in features_strings:
        features_dict = json.loads(s)
        for field in ndarray_fields:
            features_dict[field] = np.asarray(features_dict[field])

        yield DocREDInputFeatures(**features_dict)


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = setup_SSAN_parser()
    parser.add_argument("--examples_file", required=True, help="File containing serialized DocREDExample objects.")
    parser.add_argument("--features_file", required=True, help="File containing serialized DocREDInputFeatures objects.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    args.predict_thresh = 0.46544307

    logging.info(f"n_gpu: {args.n_gpu} and device: {args.device}")


    logging.info("Deserializing examples from file")
    examples_gen = deserialize_examples_file(args.examples_file)
    examples = list(examples_gen)

    logging.info("Deserializing features from file")
    features_gen = deserialize_features_file(args.features_file)
    features = list(features_gen)

    logging.info("Finished deserialization")


    ModelArch = None
    if args.model_type == 'roberta':
        ModelArch = RobertaForDocRED
    elif args.model_type == 'bert':
        ModelArch = BertForDocRED

    if args.no_naive_feature:
        with_naive_feature = False
    else:
        with_naive_feature = True


    # Set seed
    set_seed(args)

    processor = DocREDProcessor()
    label_map = processor.get_label_map(args.data_dir)
    num_labels = len(label_map.keys())


    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    # predict
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
    model = ModelArch.from_pretrained(args.checkpoint_dir,
                                      from_tf=bool(".ckpt" in args.model_name_or_path),
                                      config=config,
                                      cache_dir=args.cache_dir if args.cache_dir else None,
                                      num_labels=num_labels,
                                      max_ent_cnt=args.max_ent_cnt,
                                      with_naive_feature=with_naive_feature,
                                      entity_structure=args.entity_structure)
    model.to(args.device)

    predict_from_examples_and_features(args, model, tokenizer, examples, features)

if __name__ == "__main__":
    main()
