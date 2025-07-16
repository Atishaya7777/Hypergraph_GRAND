def model():
    '''
    Change the hidden layer size to be 4, then 8, then 16. Try all of them.
    '''
    return {}  # This should return just the latent representation


def loss_function(model_output, node_labels):
    # Calculate C_i (Clusters centroids) from your model_output
    '''
    for each cluster in your clusters:
        average the representation based on the nodes
        c_i = \sum_{v \in cluster (X_i)} v / len(c_i)

    err_i = \sum_{v \in cluster} v - c_i (Depending upon which is better, you can normalize it with respect to the cluster size X_i)
    final_error = average/sum of err_i
    '''


def error_function(model_output, correct_node_labeling_matrix):
    '''
    Basically do k means => We obtain v and i' (i' is basically the cluster index but we don't know which actual cluster index it belongs to)
    To designate which cluster the model is saying that the vertex belongs to, just simply do majority vote for that cluster and vertex pair respectively.
    - Report confusion matrix.
    '''
