"""
src/
----
Reusable modules for the Customer Segmentation project.

Modules
-------
rfm         : RFM computation, scoring, and rule-based labeling
clustering  : K-Means pipeline, scaling, evaluation, and prediction
"""

from .rfm import (
    compute_rfm,
    score_rfm,
    label_segments,
    segment_summary,
    full_rfm_pipeline,
)

from .clustering import (
    prepare_features,
    find_optimal_k,
    train_kmeans,
    predict_cluster,
    cluster_profile,
    save_model,
    load_model,
)

__all__ = [
    # RFM
    'compute_rfm',
    'score_rfm',
    'label_segments',
    'segment_summary',
    'full_rfm_pipeline',
    # Clustering
    'prepare_features',
    'find_optimal_k',
    'train_kmeans',
    'predict_cluster',
    'cluster_profile',
    'save_model',
    'load_model',
]
