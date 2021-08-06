import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def report_metrics(loss, acc, prec, recall, f1, report=print):
    report(f"[loss={loss:.3f}; acc={acc:.3f}; prec={prec:.3f}; recall={recall:.3f}; f1={f1:.3f};]")


def precision(y_true, y_pred):
    """
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision_score = true_positives / (predicted_positives + K.epsilon())

    return precision_score


def recall(y_true, y_pred):
    """
    Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_score = true_positives / (possible_positives + K.epsilon())
    return recall_score


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)


def precision_recall_f1_score(y_true, y_pred, average=None):
    assert average in (None, "binary", "macro", "micro", "weighted")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='micro')
    return precision, recall, f1


def transformers_aprf_metrics(pred):
    """
    Calculate huggingface transformers
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }