"""

Utility functions to deal with Merlin datasets in the flow when Metaflow + AWS as datastore is used.

"""

def tar_to_s3(
    folder: str,
    s3_client
    ):
    """
    Upload a folder to a tar.gz archive and return the s3 url and the local tar
    """
    import tarfile
    local_tar_name = "{}.tar.gz".format(folder)
    with tarfile.open(local_tar_name, mode="w:gz") as _tar:
        _tar.add('{}/'.format(folder), recursive=True)
    with open(local_tar_name, "rb") as in_file:
        data = in_file.read()
        url = s3_client.put(local_tar_name, data)
        print("Folder saved at: {}".format(url))

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