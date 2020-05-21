"""
Defines classic metrics, used during evaluation and testing loops.
"""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def binary_tp(gold, pred):
    r"""Get the True Positive from a list of predictions.

    For each member in pred, if it overlaps with any member of gold, return 1 else 0.

    Args:
        gold (torch.tensor): gold labels.
        pred (torch.tensor): predictions labels.

    Returns:
        int: binary true positive.

    """
    tps = 0
    for p in pred:
        tp = False
        for word in p:
            for span in gold:
                if word in span:
                    tp = True
        if tp is True:
            tps += 1
    return tps


def binary_fn(gold, pred):
    r"""Get the False Negative from a list of predictions.

    If there is any member of gold that overlaps with no member of pred, return 1 else 0.

    Args:
        gold (torch.tensor): gold labels.
        pred (torch.tensor): predictions labels.

    Returns:
        int: binary false negative.

    """
    fns = 0
    for p in gold:
        fn = True
        for word in p:
            for span in pred:
                if word in span:
                    fn = False
        if fn is True:
            fns += 1
    return fns


def binary_fp(gold, pred):
    r"""Get the False Positive from a list of predictions.

    If there is any member of pred that overlaps with no member of gold, return 1 else 0.

    Args:
        gold (torch.tensor): gold labels.
        pred (torch.tensor): predictions labels.

    Returns:
        int: binary false positive.

    """
    fps = 0
    for p in pred:
        fp = True
        for word in p:
            for span in gold:
                if word in span:
                    fp = False
        if fp is True:
            fps += 1
    return fps


def binary_precision(anns, anntype="source"):
    r"""Get the binary precision from a dictionary made of predictions and gold labels.

    Args:
        anns (dict): dictionary containing gold labels and predictions data.
        anntype (string): name of the index where list of predictions and gold labels are stored within ``anns``

    Returns:
        float: binary precision.

    """
    tps = 0
    fps = 0
    for i, ann in anns.items():
        gold = ann["gold"][anntype]
        pred = ann["pred"][anntype]
        tps += binary_tp(gold, pred)
        fps += binary_fp(gold, pred)
    return tps / (tps + fps + 10 ** -10)


def binary_recall(anns, anntype="source"):
    r"""Get the binary recall from a dictionary made of predictions and gold labels.

    Args:
        anns (dict): dictionary containing gold labels and predictions data.
        anntype (string): name of the index where list of predictions and gold labels are stored within ``anns``

    Returns:
        float: binary recall.

    """
    tps = 0
    fns = 0
    for i, ann in anns.items():
        gold = ann["gold"][anntype]
        pred = ann["pred"][anntype]
        tps += binary_tp(gold, pred)
        fns += binary_fn(gold, pred)
    return tps / (tps + fns + 10 ** -10)


def binary_f1(anns, anntype="source"):
    r"""Get the binary F1 score from a dictionary made of predictions and gold labels.

    Args:
        anns (dict): dictionary containing gold labels and predictions data.
        anntype (string): name of the index where list of predictions and gold labels are stored within ``anns``

    Returns:
        float: binary F1 score.

    """
    prec = binary_precision(anns, anntype)
    rec = binary_recall(anns, anntype)
    return 2 * ((prec * rec) / (prec + rec))


def binary_analysis(pred_analysis):
    r"""Display a binary statistical analysis of the different scores from a set of gold labels and predictions.

    Args:
        pred_analysis (dict): dictionary containing gold labels and predictions data.

    Returns:
        float: binary F1 score.

    """
    print("Binary results:")
    print("#" * 80)
    print()

    # Targets
    prec = binary_precision(pred_analysis, "target")
    rec = binary_recall(pred_analysis, "target")
    f1 = binary_f1(pred_analysis, "target")
    print("Target prec: {0:.3f}".format(prec))
    print("Target recall: {0:.3f}".format(rec))
    print("Target F1: {0:.3f}".format(f1))
    print()
    return f1


def proportional_analysis(flat_gold_labels, flat_predictions):
    r"""Display a proportional statistical analysis of the different scores from a set of gold labels and predictions.

    Args:
        flat_gold_labels (torch.tensor): list containing gold labels.
        flat_predictions (torch.tensor): list containing predictions.

    Returns:
        float: binary F1 score.

    """
    target_labels = [1, 2, 3, 4]

    print("Proportional results:")
    print("#" * 80)
    print()

    # Targets
    prec = precision_score(flat_gold_labels, flat_predictions, labels=target_labels, average="micro")
    rec = recall_score(flat_gold_labels, flat_predictions, labels=target_labels, average="micro")
    f1 = f1_score(flat_gold_labels, flat_predictions, labels=target_labels, average="micro")

    print("Target prec: {0:.3f}".format(prec))
    print("Target recall: {0:.3f}".format(rec))
    print("Target F1: {0:.3f}".format(f1))
    print()
    return f1


def get_analysis(sents, y_pred, y_test):
    pred_analysis = {}
    #
    for i, (sent, pred, gold) in enumerate(zip(sents, y_pred, y_test)):
        target = []
        #
        t = None
        # Targets
        for j, p in enumerate(pred):
            if p > 0:
                if t is None:
                    t = []
                    t.append(sent[j])
                else:
                    t.append(sent[j])
            else:
                if t is not None:
                    target.append(t)
                    t = None
        #
        gold_target = []
        #
        t = None
        # Targets
        for j, p in enumerate(gold):
            if p > 0:
                if t is None:
                    t = []
                    t.append(sent[j])
                else:
                    t.append(sent[j])
            else:
                if t is not None:
                    gold_target.append(t)
                    t = None
        #
        pred_analysis[i] = {}
        pred_analysis[i]["sent"] = [w for w in sent]
        pred_analysis[i]["gold"] = {}
        pred_analysis[i]["gold"]["target"] = gold_target
        pred_analysis[i]["pred"] = {}
        pred_analysis[i]["pred"]["target"] = target
    #
    return pred_analysis
