from collections import defaultdict, namedtuple
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


logger = logging.getLogger("selene")


Metric = namedtuple("Metric", ["fn", "data"])


def visualize_roc_curves(prediction,
                         target,
                         output_dir,
                         report_gt_feature_n_positives=50,
                         style="seaborn-colorblind",
                         fig_title="Feature ROC curves",
                         dpi=500):
    """
    Output the ROC curves for each feature predicted by a model
    as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    report_gt_feature_n_positives : int, optional
        Default is 50. Do not visualize an ROC curve for a feature with
        less than 50 positive examples in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature ROC curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """

    os.makedirs(output_dir, exist_ok=True)
    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/home/yz/.local/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf')
    fig_title1 = "DNase ROC curves"
    fig_title2 = "TF ROC curves"
    fig_title3 = "HistoneMark ROC curves"

    plt.rcParams['font.sans-serif'] = 'Arial'
    
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        if index>=0 and index<=124:
            feature_targets = target[:, index]
            if len(np.unique(feature_targets)) > 1 and \
                    np.sum(feature_targets) > report_gt_feature_n_positives:
                fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
                plt.plot(fpr, tpr)
               
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if fig_title1:
        plt.title(fig_title1)
    plt.savefig(os.path.join(output_dir, "Dnase_roc_curves.png"))
    plt.savefig(os.path.join(output_dir, "Dnase_roc_curves.svg"),format="svg",dpi=dpi)
    
    plt.clf()
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        if index>=125 and index<=814:
            feature_targets = target[:, index]
            if len(np.unique(feature_targets)) > 1 and \
                    np.sum(feature_targets) > report_gt_feature_n_positives:
                fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
                plt.plot(fpr, tpr)
               
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if fig_title2:
        plt.title(fig_title2)
    plt.savefig(os.path.join(output_dir, "TF_roc_curves.png"))
    plt.savefig(os.path.join(output_dir, "TF_roc_curves.svg"),format="svg",dpi=dpi)
    
    plt.clf()
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        if index>=815 and index<=918:
            feature_targets = target[:, index]
            if len(np.unique(feature_targets)) > 1 and \
                    np.sum(feature_targets) > report_gt_feature_n_positives:
                fpr, tpr, _ = roc_curve(feature_targets, feature_preds)
                plt.plot(fpr, tpr)
               
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if fig_title3:
        plt.title(fig_title3)
    plt.savefig(os.path.join(output_dir, "HistoneMark_roc_curves.png"))
    plt.savefig(os.path.join(output_dir, "HistoneMark_roc_curves.svg"),format="svg",dpi=dpi)
  


def visualize_precision_recall_curves(
        prediction,
        target,
        output_dir,
        report_gt_feature_n_positives=50,
        style="seaborn-colorblind",
        fig_title="Feature precision-recall curves",
        dpi=500):
    """
    Output the precision-recall (PR) curves for each feature predicted by
    a model as an SVG.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    output_dir : str
        The path to the directory to output the figures. Directories that
        do not currently exist will be automatically created.
    report_gt_feature_n_positives : int, optional
        Default is 50. Do not visualize an PR curve for a feature with
        less than 50 positive examples in `target`.
    style : str, optional
        Default is "seaborn-colorblind". Specify a style available in
        `matplotlib.pyplot.style.available` to use.
    fig_title : str, optional
        Default is "Feature precision-recall curves". Set the figure title.
    dpi : int, optional
        Default is 500. Specify dots per inch (resolution) of the figure.

    Returns
    -------
    None
        Outputs the figure in `output_dir`.

    """

    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    backend = matplotlib.get_backend()
    if "inline" not in backend:
        matplotlib.use("SVG")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/home/yz/.local/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf')
    fig_title1 = "DNase precision-recall curves"
    fig_title2 = "TF precision-recall curves"
    fig_title3 = "HistoneMark precision-recall curves"

    plt.rcParams['font.sans-serif'] = 'Arial'
    
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if index>=0 and index<=124 and len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision,
                alpha=0.3, lw=1, where="post")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if fig_title1:
        plt.title(fig_title1)
    plt.savefig(os.path.join(output_dir, "Dnase_precision_recall_curves.png"))
    plt.savefig(os.path.join(output_dir, "Dnase_precision_recall_curves.svg"),format="svg",dpi=dpi)
    
    plt.clf()
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if index>=125 and index<=814 and len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision,
                alpha=0.3, lw=1, where="post")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if fig_title2:
        plt.title(fig_title2)
    plt.savefig(os.path.join(output_dir, "TF_precision_recall_curves.png"))
    plt.savefig(os.path.join(output_dir, "TF_precision_recall_curves.svg"),format="svg",dpi=dpi)
    
    plt.clf()
    plt.style.use(style)
    plt.figure()
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if index>=815 and index<=918 and len(np.unique(feature_targets)) > 1 and \
                np.sum(feature_targets) > report_gt_feature_n_positives:
            precision, recall, _ = precision_recall_curve(
                feature_targets, feature_preds)
            plt.step(
                recall, precision,
                alpha=0.3, lw=1, where="post")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if fig_title3:
        plt.title(fig_title3)
    plt.savefig(os.path.join(output_dir, "HistoneMark_precision_recall_curves.png"))
    plt.savefig(os.path.join(output_dir, "HistoneMark_precision_recall_curves.svg"),format="svg",dpi=dpi)
    


def compute_score(prediction, target, metric_fn,
                  report_gt_feature_n_positives=10):
    """
    Using a user-specified metric, computes the distance between
    two tensors.

    Parameters
    ----------
    prediction : numpy.ndarray
        Value predicted by user model.
    target : numpy.ndarray
        True value that the user model was trying to predict.
    metric_fn : types.FunctionType
        A metric that can measure the distance between the prediction
        and target variables.
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.

    Returns
    -------
    average_score, feature_scores : tuple(float, numpy.ndarray)
        A tuple containing the average of all feature scores, and a
        vector containing the scores for each feature. If there were
        no features meeting our filtering thresholds, will return
        `(None, [])`.
    """
    feature_scores = np.ones(target.shape[1]) * np.nan
    for index, feature_preds in enumerate(prediction.T):
        feature_targets = target[:, index]
        if len(np.unique(feature_targets)) > 0 and \
               np.count_nonzero(feature_targets) > report_gt_feature_n_positives:
            try:
                feature_scores[index] = metric_fn(
                    feature_targets, feature_preds)
            except ValueError:  # do I need to make this more generic?
                continue
    valid_feature_scores = [s for s in feature_scores if not np.isnan(s)] # Allow 0 or negative values.
    if not valid_feature_scores:
        return None, feature_scores
    average_score = np.average(valid_feature_scores)
    return average_score, feature_scores


def get_feature_specific_scores(data, get_feature_from_index_fn):
    """
    Generates a dictionary mapping feature names to feature scores from
    an intermediate representation.

    Parameters
    ----------
    data : list(tuple(int, float))
        A list of tuples, where each tuple contains a feature's index
        and the score for that feature.
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).

    Returns
    -------
    dict
        A dictionary mapping feature names (`str`) to scores (`float`).
        If there was no score for a feature, its score will be set to
        `None`.

    """
    feature_score_dict = {}
    for index, score in enumerate(data):
        feature = get_feature_from_index_fn(index)
        if not np.isnan(score):
            feature_score_dict[feature] = score
        else:
            feature_score_dict[feature] = None
    return feature_score_dict


class PerformanceMetrics(object):
    """
    Tracks and calculates metrics to evaluate how closely a model's
    predictions match the true values it was designed to predict.

    Parameters
    ----------
    get_feature_from_index_fn : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    report_gt_feature_n_positives : int, optional
        Default is 10. The minimum number of positive examples for a
        feature in order to compute the score for it.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions.
        By default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, and `"average_precision"`,
        which maps to `sklearn.metrics.average_precision_score`.



    Attributes
    ----------
    skip_threshold : int
        The minimum number of positive examples of a feature that must
        be included in an update for a metric score to be
        calculated for it.
    get_feature_from_index : types.FunctionType
        A function that takes an index (`int`) and returns a feature
        name (`str`).
    metrics : dict
        A dictionary that maps metric names (`str`) to metric objects
        (`Metric`). By default, this contains `"roc_auc"` and
        `"average_precision"`.

    """

    def __init__(self,
                 get_feature_from_index_fn,
                 report_gt_feature_n_positives=10,
                 metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score)):
        """
        Creates a new object of the `PerformanceMetrics` class.
        """
        self.skip_threshold = report_gt_feature_n_positives
        self.get_feature_from_index = get_feature_from_index_fn
        self.metrics = dict()
        for k, v in metrics.items():
            self.metrics[k] = Metric(fn=v, data=[])

    def add_metric(self, name, metric_fn):
        """
        Begins tracking of the specified metric.

        Parameters
        ----------
        name : str
            The name of the metric.
        metric_fn : types.FunctionType
            A metric function.

        """
        self.metrics[name] = Metric(fn=metric_fn, data=[])

    def remove_metric(self, name):
        """
        Ends the tracking of the specified metric, and returns the
        previous scores associated with that metric.

        Parameters
        ----------
        name : str
            The name of the metric.

        Returns
        -------
        list(float)
            The list of feature-specific scores obtained by previous
            uses of the specified metric.

        """
        data = self.metrics[name].data
        del self.metrics[name]
        return data

    def update(self, prediction, target):
        """
        Evaluates the tracked metrics on a model prediction and its
        target value, and adds this to the metric histories.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.

        Returns
        -------
        dict
            A dictionary mapping each metric names (`str`) to the
            average score of that metric across all features
            (`float`).

        """
        metric_scores = {}
        for name, metric in self.metrics.items():
            avg_score, feature_scores = compute_score(
                prediction, target, metric.fn,
                report_gt_feature_n_positives=self.skip_threshold)
            metric.data.append(feature_scores)
            metric_scores[name] = avg_score
        return metric_scores

    def visualize(self, prediction, target, output_dir, **kwargs):
        """
        Outputs ROC and PR curves. Does not support other metrics
        currently.

        Parameters
        ----------
        prediction : numpy.ndarray
            Value predicted by user model.
        target : numpy.ndarray
            True value that the user model was trying to predict.
        output_dir : str
            The path to the directory to output the figures. Directories that
            do not currently exist will be automatically created.
        **kwargs : dict
            Keyword arguments to pass to each visualization function. Each
            function accepts the following args:

                * style : str - Default is "seaborn-colorblind". Specify a \
                          style available in \
                          `matplotlib.pyplot.style.available` to use.
                * dpi : int - Default is 500. Specify dots per inch \
                              (resolution) of the figure.

        Returns
        -------
        None
            Outputs figures to `output_dir`.

        """
        os.makedirs(output_dir, exist_ok=True)
        if "roc_auc" in self.metrics:
            visualize_roc_curves(
                prediction, target, output_dir,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)
        if "average_precision" in self.metrics:
            visualize_precision_recall_curves(
                prediction, target, output_dir,
                report_gt_feature_n_positives=self.skip_threshold,
                **kwargs)

    def write_feature_scores_to_file(self, output_path):
        """
        Writes each metric's score for each feature to a specified
        file.

        Parameters
        ----------
        output_path : str
            The path to the output file where performance metrics will
            be written.

        Returns
        -------
        dict
            A dictionary mapping feature names (`str`) to
            sub-dictionaries (`dict`). Each sub-dictionary then maps
            metric names (`str`) to the score for that metric on the
            given feature. If a metric was not evaluated on a given
            feature, the score will be `None`.

        """
        feature_scores = defaultdict(dict)
        for name, metric in self.metrics.items():
            feature_score_dict = get_feature_specific_scores(
                metric.data[-1], self.get_feature_from_index)
            for feature, score in feature_score_dict.items():
                if score is None:
                    feature_scores[feature] = None
                else:
                    feature_scores[feature][name] = score

        metric_cols = [m for m in self.metrics.keys()]
        cols = '\t'.join(["class"] + metric_cols)
        with open(output_path, 'w+') as file_handle:
            file_handle.write("{0}\n".format(cols))
            for feature, metric_scores in sorted(feature_scores.items()):
                if not metric_scores:
                    file_handle.write("{0}\t{1}\n".format(feature, "\t".join(["NA"] * len(metric_cols))))
                else:
                    metric_score_cols = '\t'.join(
                        ["{0:.4f}".format(s) for s in metric_scores.values()])
                    file_handle.write("{0}\t{1}\n".format(feature,
                                                          metric_score_cols))
        return feature_scores
