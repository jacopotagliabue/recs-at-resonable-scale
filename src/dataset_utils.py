"""

Utility functions to deal with Merlin datasets in the flow when Metaflow + AWS as datastore is used.

"""

def tar_to_s3(
    folder: str,
    s3_client
    ):
    """
    Upload a folder to a tar.gz archive and return the s3 url
    """
    import tarfile
    local_tar_name = "{}.tar.gz".format(folder)
    with tarfile.open(local_tar_name, mode="w:gz") as _tar:
        _tar.add('{}/'.format(folder), recursive=True)
    with open(local_tar_name, "rb") as in_file:
        data = in_file.read()
        url = s3_client.put(local_tar_name, data)
        print("Folder saved at: {}".format(url))
    # debug
    my_tar = tarfile.open(local_tar_name)
    print("Tar members: ", my_tar.getmembers())

    return url, local_tar_name

def upload_dataset_folders(
    s3_client,
    folders: list
    ):
    folders_to_s3_file = {}
    for folder in folders:
        url, local_tar_name = tar_to_s3(folder, s3_client)
        folders_to_s3_file[folder] = (url, local_tar_name)
        # delete folder locally
        import shutil
        shutil.rmtree('{}/'.format(folder))

    return folders_to_s3_file


def get_dataset_folders(
    s3_client,
    folders_to_s3_file: dict,
    target_folder: str
):
    local_paths = {}
    import tarfile  
    # getting the datasets
    for folder, (s3_file, file_name) in folders_to_s3_file.items():
        s3_obj = s3_client.get(file_name)
        local_paths[folder] = s3_obj.path
        my_tar = tarfile.open(s3_obj.path)
        my_tar.extractall(target_folder)
        my_tar.close()

    return local_paths


def prepare_predictions_for_comet_panel(
        h_m_shoppers,
        best_predictions,
        item_id_2_meta,
        api_key,
        experiment_key
    ):
        from comet_ml import ExistingExperiment
        # log some predictions as well, for the first X shoppers
        n_shoppers = 10
        predictions_to_log = []
        for shopper in h_m_shoppers[:n_shoppers]:
            cnt_predictions = best_predictions.get(shopper, None)
            # there should be preds, but check to be extra sure
            if not cnt_predictions:
                continue
            # append predictions one by one
            for p in cnt_predictions['items']:
                product_type = item_id_2_meta[p]['product_group_name'] if p in item_id_2_meta else 'NO_GROUP' 
                predictions_to_log.append({
                    "user_id": shopper,
                    "product_id": p,
                    # TODO: improve how meta-data are handled here
                    "product_type": product_type,
                    # TODO: log score from two-tower model
                    "score": 1.0
                })
        # linking prediction to the experiment for visualization
        experiment = ExistingExperiment(
            api_key=api_key,
            experiment_key= experiment_key
        )
        experiment.log_asset_data(predictions_to_log, name='predictions.json')

        return predictions_to_log