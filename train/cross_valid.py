import copy

from sklearn.model_selection import KFold, StratifiedKFold

from train.trainer import Trainer
from train.dataset import TextDataset
from util.metric import report_metrics

class CVFramework(object):
    """
    Cross validation framework to do k-fold cross validation
    """

    def __init__(self, trainer: Trainer, k=5):
        self.k = k
        self.trainers = (copy.deepcopy(trainer) for i in range(k))

    def validate(self, dataset: TextDataset):
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        loss, acc, prec, recall, f1 = .0, .0, .0, .0, .0
        for k, (train_index, eval_index) in enumerate(kf.split(dataset.get_texts(), dataset.get_labels())):

            print(f"the {k}-th fold validation.")

            train_dataset = (dataset.get_sequences_by_index(train_index), dataset.get_labels_by_index(train_index))
            eval_dataset = (dataset.get_sequences_by_index(eval_index), dataset.get_labels_by_index(eval_index))

            k_loss, k_acc, k_prec, k_recall, k_f1 = next(self.trainers).fit(train_dataset, eval_dataset)

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

