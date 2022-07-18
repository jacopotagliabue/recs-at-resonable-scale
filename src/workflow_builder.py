from pyarrow import Table

def read_to_dataframe(
    dataset: Table,
    label: str # follows naming convention train,valid,test
    ):
    import pandas as pd
    from pyarrow.parquet import write_table
    write_table(dataset, '{}.parquet'.format(label))
    return pd.read_parquet('{}.parquet'.format(label))
    

def get_nvt_workflow():
    import nvtabular.ops as ops
    import numpy as np
    import nvtabular as nvt

    user_id = ["customer_id"] >> ops.Categorify() >> ops.AddMetadata(tags=["user_id", "user"]) 
    user_features = [
        "fn",
        "active",
        "club_member_status",
        "fashion_news_frequency",        
        "postal_code"] >> ops.Categorify() >> ops.AddMetadata(tags=["user"])
    
    age_boundaries = list(np.arange(0,100,5))
    user_age = ["age"] >> ops.FillMissing(0) >> ops.Bucketize(age_boundaries) >> ops.Categorify() >> ops.AddMetadata(tags=["user"])
    user_features = user_features + user_age
    
    purchase_month = (
        ["t_dat"] >> 
        ops.LambdaOp(lambda col: col.dt.month) >> 
        ops.Rename(name ='purchase_month')
    )
    
    purchase_year = (
        ["t_dat"] >> 
        ops.LambdaOp(lambda col: col.dt.year) >> 
        ops.Rename(name ='purchase_year')
    )
    
    context_features = (
        (purchase_month + purchase_year) >> ops.Categorify() >>  ops.AddMetadata(tags=["user"])     
    )
        
    item_id = ["article_id"] >> ops.Categorify() >> ops.AddMetadata(tags=["item_id", "item"]) 
    item_features = ["product_code", 
            "product_type_no", 
            "product_group_name", 
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
            "department_no",
            "index_code",
            "index_group_no",
            "section_no",
            "garment_group_no"] >> ops.Categorify() >>  ops.AddMetadata(tags=["item"]) 
    
    item_avg_price = (
        item_id >> ops.JoinGroupby(
            cont_cols=["price"],
            stats=["mean"]
        ) >>
        ops.FillMissing(0) >>
        ops.Normalize() >>
        ops.Rename(name = "avg_price") >>
        ops.AddMetadata(tags=["item"])
    )    
    
    item_features = item_features + item_avg_price

    outputs = user_id + user_features + context_features + item_id + item_features

    workflow = nvt.Workflow(outputs)

    return workflow