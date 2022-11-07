"""

Utility functions to deal with Merlin models.

"""
from merlin.io.dataset import Dataset 


def get_items_topk_recommender_model(
    train_dataset: Dataset, 
    schema, 
    model, 
    k: int
    ):
    from merlin.schema.tags import Tags
    item_features = schema.select_by_tag(Tags.ITEM).column_names
    item_dataset = train_dataset.to_ddf()[item_features].drop_duplicates(subset=['article_id'], keep='last').compute()
    item_dataset = Dataset(item_dataset)

    return model.to_top_k_recommender(item_dataset, k=k)
