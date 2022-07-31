import streamlit as st
import pandas as pd
import numpy as np
from metaflow import Flow, Step
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


# some global variables
FLOW_NAME = 'merlinFlow' # make sure it's the same as my_merlin_flow.py
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_html_image_tag(image_url: str):
    """
    
    Trick from https://github.com/streamlit/streamlit/issues/1873

    """
    return '<img src="{}" height="100">'.format(image_url)


@st.cache
def get_artifacts_from_last_run(flow_name: str):
    """
    User Metaflow API to get artifacts from last successful run.

    See: https://docs.metaflow.org/metaflow/client
    """
    flow = Flow(flow_name)
    target_run_id = flow.latest_successful_run.id
    print("Run Id: {}".format(target_run_id))
    target_step = Step('{}/{}/export_to_app'.format(flow_name, target_run_id))
    # retrieve the dataframe including metadata and the CLIP image vectors
    df = target_step.task.data.prediction_df
    print("===== Df loaded from Metaflow =====\n")
    print(df.head(3))
    df['target_image'] = df.apply(lambda row: build_html_image_tag(row['target_image_url']), axis=1)
    df['predicted_image'] = df.apply(lambda row: build_html_image_tag(row['predicted_image_url']), axis=1)

    return df


# Utility functions to use CLIP


def load_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor, tokenizer


def encode_text(text, model, tokenizer, processor):
    """
    
    Encode text from: https://huggingface.co/spaces/DrishtiSharma/Image-search-using-CLIP/blob/main/app.py
    
    """
    with torch.no_grad():
        inputs = tokenizer([text], padding=True, return_tensors="pt")
        inputs = processor(text=[text], images=None, return_tensors="pt", padding=True)
        text_feature = model.get_text_features(**inputs)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
    
    return text_feature.detach().numpy()


# App-specific code


# load data from metaflow and the CLIP model
metaflow_df = get_artifacts_from_last_run(FLOW_NAME)
model, processor, tokenizer = load_clip(device)

# get unique product types
product_types = [None] + list(metaflow_df['product_type'].unique())

# decide which columns to show
cols = ['target_item', 'target_image', 'product_type', 'predicted_item', 'predicted_image']

# app title
st.title('Inspecting recommendations made by the model')

# show data from Metaflow
if st.checkbox('Show raw data from Metaflow'):
    st.write(metaflow_df[:3])

st.header('Display predictions')
option = st.selectbox('Filter by product type', product_types)
st.write('You selected: `{}`'.format(option))

query = st.text_input('Free-text search', '')
st.write('You searched for: `{}`'.format(query))

df = metaflow_df.copy()
# if there is a filter specified, filter the dataframe
if option is not None:
    df = df[df['product_type'] == option]

# if there is a text specified, sort the dataframe
if query.strip() != '':
    encoded_text = encode_text(query.strip().lower(), model, tokenizer, processor)
    # get all image vectors in numpy array
    image_vectors = np.array(list(df['image_vectors']))
    df['dot_product'] = list((encoded_text @ image_vectors.T).squeeze(0))
    df = df.sort_values(by='dot_product', ascending=False)

# diplay the dataframe as HTML to have images
st.write(df[cols].to_html(escape=False), unsafe_allow_html=True)