from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import collections

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

import models.tf.nlp.bert.modeling as modeling
import models.tf.nlp.bert.optimization as optimization
import models.tf.nlp.bert.tokenization as tokenization



tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def get_metrics_ops(labels, predictions, num_labels):
    cm, op = _streaming_confusion_matrix(labels, predictions, num_labels)
    tf.logging.info(type(cm))
    tf.logging.info(type(op))
    return (tf.convert_to_tensor(cm), op)


def get_metrics(conf_mat, num_labels):
    precisions = []
    recalls = []
    for i in range(num_labels):
        tp = conf_mat[i][i].sum()
        col_sum = conf_mat[:, i].sum()
        row_sum = conf_mat[i].sum()

        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    pre = sum(precisions) / len(precisions)
    rec = sum(recalls) / len(recalls)
    f1 = 2 * pre * rec / (pre + rec)

    return pre, rec, f1


class Bert(object):

    def __init__(self,
                 bert_config_file,
                 vocab_file,
                 init_checkpoint,
                 do_lower_case=True,
                 max_seq_length=128
                 ):

        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.vocab_file = vocab_file
        self.init_checkpoint = init_checkpoint

        self.do_lower_case = do_lower_case
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.max_seq_length = max_seq_length

        model_fn = self.model_fn_builder()

        config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True, gpu_options={"allow_growth": True})

        run_config = tf.estimator.RunConfig(
            session_config=config,
            save_checkpoints_steps=0)

        self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": 64})

        self.use_one_hot_embeddings = False

    def input_fn_builder(self, features):
        pass

    def model_fn_builder(self):
        pass


class BertFeatureExtractor(Bert):
    """

    """

    class FeatureExtractorInputSample(object):
        def __init__(self, unique_id, text_a, text_b):
            self.unique_id = unique_id
            self.text_a = text_a
            self.text_b = text_b

    class FeatureExtractorInputFeatures(object):
        """
        A single set of feature of data.
        """

        def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __init__(self,
                 bert_config_file,
                 vocab_file,
                 init_checkpoint,
                 do_lower_case,
                 max_seq_length):
        Bert.__init__(self, bert_config_file, vocab_file, init_checkpoint, do_lower_case, max_seq_length)


    @staticmethod
    def create_input(texts):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        unique_id = 0
        for text in texts:
            line = tokenization.convert_to_unicode(text)
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                BertFeatureExtractor.FeatureExtractorInputSample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
        return examples

    @staticmethod
    def convert_examples_to_features(examples, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                BertFeatureExtractor._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the train to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire train is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append("[SEP]")
                input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            features.append(
                BertFeatureExtractor.FeatureExtractorInputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features

    def input_fn_builder(self, features):
        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []

        for feature in features:
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            d = tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                    tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, self.max_seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, self.max_seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.constant(
                        all_input_type_ids,
                        shape=[num_examples, self.max_seq_length],
                        dtype=tf.int32),
            })

            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d

        return input_fn

    def model_fn_builder(self):

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            unique_ids = features["unique_ids"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_type_ids = features["input_type_ids"]

            model = modeling.BertModel(
                config=self.bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids)

            if mode != tf.estimator.ModeKeys.PREDICT:
                raise ValueError("Only PREDICT modes are supported: %s" % (mode))

            tvars = tf.trainable_variables()
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.init_checkpoint)

            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            all_layers = model.get_all_encoder_layers()

            predictions = {
                "unique_id": unique_ids,
            }

            # for (i, layer_index) in enumerate(self.layer_indexes):
            # output all layers
            for layer_index in range(len(all_layers)):
                predictions["layer_output_%d" % layer_index] = all_layers[layer_index]

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)
            return output_spec

        return model_fn

    def extract_feature(self, texts, layers='-1,-2,-3,-4'):

        layer_indexes = [int(x) for x in layers.split(",")]

        examples = BertFeatureExtractor.create_input(texts)

        features = BertFeatureExtractor.convert_examples_to_features(
            examples=examples, seq_length=self.max_seq_length, tokenizer=self.tokenizer)

        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

        input_fn = self.input_fn_builder(features=features)

        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                    ]
                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
                print(features)



class BertTrainer(Bert):

    class TrainInputSample(object):
        """A single training/test example for simple sequence classification."""

        def __init__(self, guid, text_a, text_b=None, label=None):
            """Constructs a InputExample.

            Args:
                guid: Unique id for the example.
                text_a: string. The untokenized text of the first sequence. For single
                    sequence tasks, only this sequence must be specified.
                text_b: (Optional) string. The untokenized text of the second sequence.
                    Only must be specified for sequence pair tasks.
                label: (Optional) string. The label of the example. This should be
                    specified for train and dev examples, but not for test examples.
            """
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label

    class TrainInputFeatures(object):
        """A single set of feature of data."""

        def __init__(self, input_ids, input_mask, segment_ids, label_id):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_id = label_id


    class TextClsData:

        def __init__(self, xs_train, ys_train, xs_dev=None, ys_dev=None):

            le = LabelEncoder()
            encoded_labels = le.fit_transform(ys_train)
            self.encoded_labels = np.unique(encoded_labels).astype("str")
            self.train_examples = []

            for i in range(len(xs_train)):
                guid = "%s-%s" % ("train", i)
                text = tokenization.convert_to_unicode(xs_train[i])
                label = tokenization.convert_to_unicode(ys_train[i])
                self.train_examples.append(
                    BertTrainer.TrainInputSample(guid=guid, text_a=text, text_b=None, label=label))

            self.dev_examples = []
            if xs_dev is not None and ys_dev is not None:
                for i in range(len(xs_dev)):
                    guid = "%s-%s" % ("train", i)
                    text = tokenization.convert_to_unicode(xs_dev[i])
                    label = tokenization.convert_to_unicode(ys_dev[i])
                    self.dev_examples.append(
                        BertTrainer.TrainInputSample(guid=guid, text_a=text, text_b=None, label=label))

        def get_train_data(self):
            return self.train_examples

        def get_dev_data(self):
            return self.dev_examples

        def get_labels(self):
            return self.encoded_labels

    class DataFrameSingleTextInputProcessor:
        def __init__(self, id_col_name, text_col_name, label_col_name):
            self.id_col_name = id_col_name
            self.text_col_name = text_col_name
            self.label_col_name = label_col_name

        def process(self, df_row):
            text_a = tokenization.convert_to_unicode(df_row[self.text_col_name])
            label = tokenization.convert_to_unicode(df_row[self.label_col_name])
            return BertTrainer.TrainInputSample(guid=id, text_a=text_a, text_b=None, label=label)

    class DataFrameTextPairInputProcessor:

        def __init__(self, id_col_name, text_a_col_name, text_b_col_name, label_col_name):
            self.id_col_name = id_col_name
            self.text_a_col_name = text_a_col_name
            self.text_b_col_name = text_b_col_name
            self.label_col_name = label_col_name

        def process(self, df_row):
            text_a = tokenization.convert_to_unicode(df_row[self.text_a_col_name])
            text_b = tokenization.convert_to_unicode(df_row[self.text_b_col_name])
            label = tokenization.convert_to_unicode(df_row[self.label_col_name])
            return BertTrainer.TrainInputSample(guid=id, text_a=text_a, text_b=text_b, label=label)


    class DataFrameInput:

        def __init__(self, input_processor, labels, train_df, dev_df):

            self.labels = labels
            self.train_examples = []

            for i, row in train_df.iterrows():
                self.train_examples.append(input_processor.process(row))

            self.dev_examples = []
            if dev_df is not None:
                for i, row in dev_df.iterrows():
                    self.dev_examples.append(input_processor.process(row))

        def get_train_data(self):
            return self.train_examples

        def get_dev_data(self):
            return self.dev_examples

        def get_labels(self):
            return self.labels


    def __init__(self,
                 bert_config_file,
                 vocab_file,
                 init_checkpoint,
                 data_dir,
                 output_dir,
                 processor,
                 do_train=True,
                 do_eval=True,
                 do_lower_case=True,
                 max_seq_length=128,
                 save_checkpoints_steps=1000,
                 train_batch_size=8,
                 eval_batch_size=8,
                 predict_batch_size=8,
                 num_train_epochs=5,
                 warmup_proportion=0.1,
                 learning_rate=5e-5
                 ):

        Bert.__init__(self,  bert_config_file, vocab_file, init_checkpoint, do_lower_case, max_seq_length)

        self.do_train = do_train
        self.do_eval = do_eval

        self.processor = processor

        self.label_list = processor.get_labels()

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.save_checkpoints_steps = save_checkpoints_steps
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.learning_rate = learning_rate

    def convert_single_example(self, ex_index, example, label_list, max_seq_length,
                               tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            """Truncates a sequence pair in place to the maximum length."""

            # This is a simple heuristic which will always truncate the longer sequence
            # one token at a time. This makes more sense than truncating an equal percent
            # of tokens from each, since if one sequence is very short then each token
            # that's truncated likely contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()


        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the train to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire train is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        feature = BertTrainer.TrainInputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
        return feature

    def create_model(self, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels):
        """Creates a classification train."""

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use train.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    def file_based_convert_examples_to_features(self, examples, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example, self.label_list,
                                             self.max_seq_length, self.tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


    def input_fn_builder(self, input_file, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        name_to_features = {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn


    def model_fn_builder(self):

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = self.create_model(
                is_training, input_ids, input_mask, segment_ids, label_ids,
                self.num_labels)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if self.init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)

                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                    total_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits):
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(label_ids, predictions)
                    loss = tf.metrics.mean(per_example_loss)
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                    }

                eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss)
                    # ,
                    # eval_metric_ops=get_metrics_ops)
            else:
                output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=probabilities)
            return output_spec

        return model_fn

    def train(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        if not self.do_train and not self.do_eval and not self.do_predict:
            raise ValueError(
                "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT train "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(self.output_dir)

        self.num_labels = len(self.label_list)

        train_examples = None
        self.num_train_steps = None

        if self.do_train:
            train_examples = self.processor.get_train_data()
            self.num_train_steps = int(
                len(train_examples) / self.train_batch_size * self.num_train_epochs)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        if self.do_train:
            train_file = os.path.join(self.output_dir, "train.tf_record")
            self.file_based_convert_examples_to_features(train_examples, train_file)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", self.train_batch_size)
            tf.logging.info("  Num steps = %d", self.num_train_steps)
            train_input_fn = self.input_fn_builder(
                input_file=train_file,
                is_training=True,
                drop_remainder=True)
            self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

        if self.do_eval:
            eval_examples = self.processor.get_dev_data()
            eval_file = os.path.join(self.output_dir, "eval.tf_record")
            self.file_based_convert_examples_to_features(eval_examples, eval_file)

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d", len(eval_examples))
            tf.logging.info("  Batch size = %d", self.eval_batch_size)


            eval_input_fn = self.input_fn_builder(
                input_file=eval_file,
                is_training=False, drop_remainder=False)

            result = self.estimator.evaluate(input_fn=eval_input_fn)

            output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))



    def predict(self):

        predict_examples = self.processor.get_test_data()
        predict_file = os.path.join(self.output_dir, "predict.tf_record")
        self.file_based_convert_examples_to_features(predict_examples, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", self.predict_batch_size)

        predict_input_fn = self.input_fn_builder(
            input_file=predict_file,
            is_training=False,
            drop_remainder=False)

        result = self.estimator.predict(input_fn=predict_input_fn)

        return result

        # output_predict_file = os.path.join(self.output_dir, "test_results.tsv")
        # with tf.gfile.GFile(output_predict_file, "w") as writer:
        #     tf.logging.info("***** Predict results *****")
        #     for prediction in result:
        #         output_line = "\t".join(
        #             str(class_probability) for class_probability in prediction) + "\n"
        #         writer.write(output_line)

    def save_model(self):
        pass


if __name__ == '__main__':

    # bert = BertFeatureExtractor(
    #     r'C:\Users\charlie\developer\train\nlp\lm\chinese_L-12_H-768_A-12\bert_config.json',
    #     r'C:\Users\charlie\developer\train\nlp\lm\chinese_L-12_H-768_A-12\vocab.txt',
    #     r'C:\Users\charlie\developer\train\nlp\lm\chinese_L-12_H-768_A-12\bert_model.ckpt',
    #     True, 128)
    #
    # bert.extract_feature([u"你说啥"])

    BERT_HOME = r"C:\Users\Charlie\Developer\models\chinese_L-12_H-768_A-12"
    # BERT_HOME = r"C:\Users\Charlie\Developer\train\multi_cased_L-12_H-768_A-12"

    DATA_HOME = r"C:\Users\Charlie\Corpus\kaggle\nlp-getting-started"

    import pandas as pd
    train_pd = pd.read_csv(os.path.join(DATA_HOME, "train.csv"))
    xs_train, ys_train = train_pd["text"], train_pd["target"].apply(lambda x: str(x))

    test_pd = pd.read_csv(os.path.join(DATA_HOME, "test.csv"))
    xs_test, ys_test = train_pd["text"], train_pd["target"].apply(lambda x: str(x))

    bert = BertTrainer(
        os.path.join(BERT_HOME, 'bert_config.json'),
        os.path.join(BERT_HOME, 'vocab.txt'),
        os.path.join(BERT_HOME, 'bert_model.ckpt'),
        DATA_HOME,
        os.path.join(DATA_HOME, 'output'),
        BertTrainer.TextClsData(xs_train, ys_train)
    )

    bert.train()