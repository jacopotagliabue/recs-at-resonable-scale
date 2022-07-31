"""

Utility functions to use CLIP in the Streamlit app. Check the README for details.

"""


# Function that computes the feature vectors for an image
# TODO: optimize this function
def encode_image(model, processor, image_url: str, device='cpu'):
    """
    From https://huggingface.co/spaces/DrishtiSharma/Image-search-using-CLIP/blob/main/app.py
    """
    import requests
    from io import BytesIO
    import torch
    from PIL import Image # pylint: disable=import-error
    # get image from url
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    with torch.no_grad():
        photo_preprocessed = processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"]
        search_photo_feature = model.get_image_features(photo_preprocessed.to(device))
        search_photo_feature /= search_photo_feature.norm(dim=-1, keepdim=True)
    
    return search_photo_feature.cpu().numpy()