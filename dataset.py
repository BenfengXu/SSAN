# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from dataclasses import dataclass
from typing import Optional
import json
import copy
import numpy


logger = logging.getLogger(__name__)


def norm_mask(input_mask):
    output_mask = numpy.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not numpy.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask


def docred_convert_examples_to_features(
    examples,
    model_type,
    tokenizer,
    max_length=512,
    max_ent_cnt=42,
    label_map=None,
    pad_token=0,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    features = []

    ner_map = {'PAD':0, 'ORG':1, 'LOC':2, 'NUM':3, 'TIME':4, 'MISC':5, 'PER':6}
    distance_buckets = numpy.zeros((512), dtype='int64')
    distance_buckets[1] = 1
    distance_buckets[2:] = 2
    distance_buckets[4:] = 3
    distance_buckets[8:] = 4
    distance_buckets[16:] = 5
    distance_buckets[32:] = 6
    distance_buckets[64:] = 7
    distance_buckets[128:] = 8
    distance_buckets[256:] = 9

    for (ex_index, example) in enumerate(examples):

        len_examples = len(examples)

        if ex_index % 500 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        input_tokens = []
        tok_to_sent = []
        tok_to_word = []
        for sent_idx, sent in enumerate(example.sents):
            for word_idx, word in enumerate(sent):
                tokens_tmp = tokenizer.tokenize(word, add_prefix_space=True)
                input_tokens += tokens_tmp
                tok_to_sent += [sent_idx] * len(tokens_tmp)
                tok_to_word += [word_idx] * len(tokens_tmp)

        if len(input_tokens) <= max_length - 2:
            if model_type == 'roberta':
                input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
            else:
                input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
            tok_to_sent = [None] + tok_to_sent + [None]
            tok_to_word = [None] + tok_to_word + [None]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            # padding
            padding = [None] * (max_length - len(input_ids))
            tok_to_sent += padding
            tok_to_word += padding
            padding = [0] * (max_length - len(input_ids))
            attention_mask += padding
            token_type_ids += padding
            padding = [pad_token] * (max_length - len(input_ids))
            input_ids += padding
        else:
            input_tokens = input_tokens[:max_length - 2]
            tok_to_sent = tok_to_sent[:max_length - 2]
            tok_to_word = tok_to_word[:max_length - 2]
            if model_type == 'roberta':
                input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
            else:
                input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
            tok_to_sent = [None] + tok_to_sent + [None]
            tok_to_word = [None] + tok_to_word + [None]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [pad_token] * len(input_ids)

        # ent_mask & ner / coreference feature
        ent_mask = numpy.zeros((max_ent_cnt, max_length), dtype='int64')
        ent_ner = [0] * max_length
        ent_pos = [0] * max_length
        tok_to_ent = [-1] * max_length
        ents = example.vertexSet
        for ent_idx, ent in enumerate(ents):
            for mention in ent:
                for tok_idx in range(len(input_ids)):
                    if tok_to_sent[tok_idx] == mention['sent_id'] \
                            and mention['pos'][0] <= tok_to_word[tok_idx] < mention['pos'][1]:
                        ent_mask[ent_idx][tok_idx] = 1
                        ent_ner[tok_idx] = ner_map[ent[0]['type']]
                        ent_pos[tok_idx] = ent_idx + 1
                        tok_to_ent[tok_idx] = ent_idx

        # distance feature
        ent_first_appearance = [0] * max_ent_cnt
        ent_distance = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='int8')  # padding id is 10
        for i in range(len(ents)):
            if numpy.all(ent_mask[i] == 0):
                continue
            else:
                ent_first_appearance[i] = numpy.where(ent_mask[i] == 1)[0][0]
        for i in range(len(ents)):
            for j in range(len(ents)):
                if ent_first_appearance[i] != 0 and ent_first_appearance[j] != 0:
                    if ent_first_appearance[i] >= ent_first_appearance[j]:
                        ent_distance[i][j] = distance_buckets[ent_first_appearance[i] - ent_first_appearance[j]]
                    else:
                        ent_distance[i][j] = - distance_buckets[- ent_first_appearance[i] + ent_first_appearance[j]]
        ent_distance += 10  # norm from [-9, 9] to [1, 19]

        structure_mask = numpy.zeros((5, max_length, max_length), dtype='float')
        for i in range(max_length):
            if attention_mask[i] == 0:
                break
            else:
                if tok_to_ent[i] != -1:
                    for j in range(max_length):
                        if tok_to_sent[j] is None:
                            continue
                        # intra
                        if tok_to_sent[j] == tok_to_sent[i]:
                            # intra-coref
                            if tok_to_ent[j] == tok_to_ent[i]:
                                structure_mask[0][i][j] = 1
                            # intra-relate
                            elif tok_to_ent[j] != -1:
                                structure_mask[1][i][j] = 1
                            # intra-NA
                            else:
                                structure_mask[2][i][j] = 1
                        # inter
                        else:
                            # inter-coref
                            if tok_to_ent[j] == tok_to_ent[i]:
                                structure_mask[3][i][j] = 1
                            # inter-relate
                            elif tok_to_ent[j] != -1:
                                structure_mask[4][i][j] = 1

        # label
        label_ids = numpy.zeros((max_ent_cnt, max_ent_cnt, len(label_map.keys())), dtype='bool')
        # test file does not have "labels"
        if example.labels is not None:
            labels = example.labels
            for label in labels:
                label_ids[label['h']][label['t']][label_map[label['r']]] = 1
        for h in range(len(ents)):
            for t in range(len(ents)):
                if numpy.all(label_ids[h][t] == 0):
                    label_ids[h][t][0] = 1

        label_mask = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
        label_mask[:len(ents), :len(ents)] = 1
        for ent in range(len(ents)):
            label_mask[ent][ent] = 0
        for ent in range(len(ents)):
            if numpy.all(ent_mask[ent] == 0):
                label_mask[ent, :] = 0
                label_mask[:, ent] = 0

        ent_mask = norm_mask(ent_mask)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        assert ent_mask.shape == (max_ent_cnt, max_length)
        assert label_ids.shape == (max_ent_cnt, max_ent_cnt, len(label_map.keys()))
        assert label_mask.shape == (max_ent_cnt, max_ent_cnt)
        assert len(ent_ner) == max_length
        assert len(ent_pos) == max_length
        assert ent_distance.shape == (max_ent_cnt, max_ent_cnt)
        assert structure_mask.shape == (5, max_length, max_length)

        if ex_index == 42:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("doc: %s" % (' '.join([' '.join(sent) for sent in example.sents])))
            logger.info("input_ids: %s" % (" ".join([str(x) for x in input_ids])))
            logger.info("attention_mask: %s" % (" ".join([str(x) for x in attention_mask])))
            logger.info("token_type_ids: %s" % (" ".join([str(x) for x in token_type_ids])))
            logger.info("ent_mask for first ent: %s" % (" ".join([str(x) for x in ent_mask[0]])))
            logger.info("label for ent pair 0-1: %s" % (" ".join([str(x) for x in label_ids[0][1]])))
            logger.info("label_mask for first ent: %s" % (" ".join([str(x) for x in label_mask[0]])))
            logger.info("ent_ner: %s" % (" ".join([str(x) for x in ent_ner])))
            logger.info("ent_pos: %s" % (" ".join([str(x) for x in ent_pos])))
            logger.info("ent_distance for first ent: %s" % (" ".join([str(x) for x in ent_distance[0]])))

        features.append(
            DocREDInputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                ent_mask=ent_mask,
                ent_ner=ent_ner,
                ent_pos=ent_pos,
                ent_distance=ent_distance,
                structure_mask=structure_mask,
                label=label_ids,
                label_mask=label_mask,
            )
        )

    return features


class DocREDProcessor(object):
    """Processor for the DocRED data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return DocREDExample(
            tensor_dict["guid"].numpy(),
            tensor_dict["title"].numpy(),
            tensor_dict["vertexSet"].numpy(),
            tensor_dict["sents"].numpy(),
            tensor_dict["labels"].numpy(),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train_annotated.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'train')

    def get_distant_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train_distant.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "dev.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "test.json"), 'r') as f:
            examples = json.load(f)
        return self._create_examples(examples, 'test')

    def get_label_map(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "label_map.json"), 'r') as f:
            label_map = json.load(f)
        return label_map

    def _create_examples(self, instances, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ins) in enumerate(instances):
            guid = "%s-%s" % (set_type, i)
            examples.append(DocREDExample(guid=guid,
                                          title=ins['title'],
                                          vertexSet=ins['vertexSet'],
                                          sents=ins['sents'],
                                          labels=ins['labels'] if set_type!="test" else None))
        return examples


@dataclass(frozen=False)
class DocREDExample:

    guid: str
    title: str
    vertexSet: list
    sents: list
    labels: Optional[list] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"


class DocREDInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, ent_mask, ent_ner, ent_pos, ent_distance, structure_mask, label=None, label_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent_mask = ent_mask
        self.ent_ner = ent_ner
        self.ent_pos = ent_pos
        self.ent_distance = ent_distance
        self.structure_mask = structure_mask
        self.label = label
        self.label_mask = label_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"