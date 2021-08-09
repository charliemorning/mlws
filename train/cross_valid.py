import copy
from collections import Counter

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from train.trainer import Trainer
from train.dataset import TextDataset
from util.metric import report_metrics
from util.math import softmax


class CVFramework(object):
    """
    Cross validation framework to do k-fold cross validation
    """

    def __init__(self, trainer: Trainer, k=5):
        self.k = k
        self.trainers = [copy.deepcopy(trainer) for i in range(k)]

    def validate(self, dataset: TextDataset) -> (float, float, float, float, float):
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        loss, acc, prec, recall, f1 = .0, .0, .0, .0, .0
        for k, (train_index, eval_index) in enumerate(kf.split(dataset.get_texts(), dataset.get_labels())):

            print(f"The {k}-th fold validation.")

            train_dataset = (dataset.get_sequences_by_index(train_index), dataset.get_labels_by_index(train_index))
            eval_dataset = (dataset.get_sequences_by_index(eval_index), dataset.get_labels_by_index(eval_index))

            k_loss, k_acc, k_prec, k_recall, k_f1 = self.trainers[k].fit(train_dataset, eval_dataset)

            loss += k_loss
            acc += k_acc
            prec += k_prec
            recall += k_recall
            f1 += k_f1

        loss /= self.k
        acc /= self.k
        prec /= self.k
        recall /= self.k
        f1 /= self.k

        print("final result:")
        report_metrics(loss, acc, prec, recall, f1)

        return loss, acc, prec, recall, f1

    def predict(self, test_data: TextDataset) -> np.array[int]:

        logits = np.zeros((len(test_data), test_data.get_label_size()))

        stack_preds = np.zeros((5, len(test_data)))
        preds = np.zeros(len(test_data))

        xs_test = test_data.get_sequences()
        for i, trainer in enumerate(self.trainers):
            k_logits, k_preds = trainer.predict(xs_test)
            stack_preds[i] = k_preds
            logits += k_logits

        stack_preds = stack_preds.transpose()
        for i in range(len(stack_preds)):
            pred = stack_preds[i]
            label_counter = Counter(pred)
            preds[i] = label_counter.most_common(1)[0][0]


        # logits /= self.k
        # preds = np.argmax(softmax(logits, axis=1), axis=1).flatten()
        return preds

    def get_trainers(self) -> list:
        return self.trainers
