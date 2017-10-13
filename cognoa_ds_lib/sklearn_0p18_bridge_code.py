import numpy as np
from scipy.sparse import coo_matrix


#### This is literally copied from the sklearn 0.18 source code
#### 0.17 does not support sample_weights, so this can be deprecated
#### Once we upgrade versions
def confusion_matrix_0p18(y_true, y_pred, labels=None, sample_weight=None):
    """Compute confusion matrix to evaluate the accuracy of a classification
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.
    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    Read more in the :ref:`User Guide <confusion_matrix>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix
    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <https://en.wikipedia.org/wiki/Confusion_matrix>`_
    Examples
    --------


    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    """

    ######### Attempt at getting sklearn's data input check working
	######### But failed due to too many missing dependencies
    ##### This is supporting code for getting the confusion matrix with weights.
    ##### literally copied from the sklearn 0.18 source code
    ##### 0.17 does not support sample_weights, so this can be deprecated
    ##### Once we upgrade versions
    #def _check_targets(y_true, y_pred):
    #    """Check that y_true and y_pred belong to the same classification task
    #    This converts multiclass or binary types to a common shape, and raises a
    #    ValueError for a mix of multilabel and multiclass targets, a mix of
    #    multilabel formats, for the presence of continuous-valued or multioutput
    #    targets, or for targets of different lengths.
    #    Column vectors are squeezed to 1d, while multilabel formats are returned
    #    as CSR sparse label indicators.
    #    Parameters
    #    ----------
    #    y_true : array-like
    #    y_pred : array-like
    #    Returns
    #    -------
    #    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
    #        The type of the true target data, as output by
    #        ``utils.multiclass.type_of_target``
    #    y_true : array or indicator matrix
    #    y_pred : array or indicator matrix
    #    """
    #    #### check_consistent_length(y_true, y_pred)
    #    
    #    type_true = type_of_target(y_true)
    #    type_pred = type_of_target(y_pred)
    #
    #    y_type = set([type_true, type_pred])
    #    if y_type == set(["binary", "multiclass"]):
    #        y_type = set(["multiclass"])
    #
    #    if len(y_type) > 1:
    #        raise ValueError("Can't handle mix of {0} and {1}"
    #                         "".format(type_true, type_pred))
    #
    #    # We can't have more than one value on y_type => The set is no more needed
    #    y_type = y_type.pop()
    #
    #    # No metrics support "multiclass-multioutput" format
    #    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
    #        raise ValueError("{0} is not supported".format(y_type))
    #
    #    if y_type in ["binary", "multiclass"]:
    #        y_true = column_or_1d(y_true)
    #        y_pred = column_or_1d(y_pred)
    #
    #    if y_type.startswith('multilabel'):
    #        y_true = csr_matrix(y_true)
    #        y_pred = csr_matrix(y_pred)
    #        y_type = 'multilabel-indicator'
    #
    #    return y_type, y_true, y_pred

    assert sample_weight is None or len(sample_weight) == len(y_true)
    assert len(y_true) == len(y_pred)
#    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
#    if y_type not in ("binary", "multiclass"):
#        raise ValueError("%s is not supported" % y_type)

#    print 'labels: ', labels
#    print 'type: ', type(labels)
#    print 'y_true: ', y_true
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
#        print 'labels: ', labels
#        print 'type: ', type(labels)
#        print 'labels.values: ', labels.values
        try:
#            print 'y_true: ', y_true, ', type: ', type(y_true)
            y_true = y_true.values
#            print 'converted to an array'
        except:
            ### Was not a pandas series
            pass
        y_true = np.array(y_true)
        labels = np.asarray(labels)
        if np.all([l not in y_true for l in labels]):
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int)
    else:
        sample_weight = np.asarray(sample_weight)

    #check_consistent_length(sample_weight, y_true, y_pred)

    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    CM = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels)
                    ).toarray()

    return CM
