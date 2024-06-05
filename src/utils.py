from seqeval import metrics



def get_metric(labels_list, preds_list, lengths_list, id2label):
    y_preds = []
    y_labels = []
    if not isinstance(labels_list, list):
        labels_list = [labels_list]
        preds_list = [preds_list]
        lengths_list = [lengths_list]
    for labels, preds, lengths in zip(labels_list, preds_list, lengths_list):
        for label, pred, l in zip(labels, preds, lengths):
            y_preds.append([id2label[pi] for pi in pred[:l]])
            y_labels.append([id2label[li] for li in label[:l]])
    return {
        "f1": metrics.f1_score(y_labels, y_preds),
        "precision": metrics.precision_score(y_labels, y_preds),
        "recall": metrics.recall_score(y_labels, y_preds)
    }

