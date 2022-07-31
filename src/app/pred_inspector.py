import streamlit as st
import pandas as pd
import numpy as np
from metaflow import Flow, Step
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


# some global variables
FLOW_NAME = 'merlinFlow' # make sure it's the same as my_merlin_flow.py
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache
def get_artifacts_from_last_run(flow_name: str):
    """
    User Metaflow API to get artifacts from last successful run.

    See: https://docs.metaflow.org/metaflow/client
    """
    flow = Flow(flow_name)
    target_run_id = flow.latest_successful_run.id
    print(target_run_id)
    target_step = Step('{}/{}/export_to_app'.format(flow_name, target_run_id))
    # retrieve the dataframe including metadata and the CLIP image vectors
    df = target_step.task.data.prediction_df
    print("===== Df loaded from Metaflow =====\n")
    print(df.head(3))

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
        inputs = tokenizer([text],  padding=True, return_tensors="pt")
        inputs = processor(text=[text], images=None, return_tensors="pt", padding=True)
    
    return model.get_text_features(**inputs).detach().numpy()


def find_best_matches(image, mode, text):
    text_features = encode_text(text)
    
    return None


# App-specific code


# load data from metaflow and the CLIP model
metaflow_df = get_artifacts_from_last_run(FLOW_NAME)
model, processor, tokenizer = load_clip(device)

# app title
st.title('Inspecting recommendations made by the model')

# show data from Metaflow
if st.checkbox('Show raw data'):
    st.subheader('Raw data from Metaflow')
    st.write(metaflow_df[:10])