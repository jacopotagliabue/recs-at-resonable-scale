# recs-at-resonable-scale
Recommendations at "Reasonable Scale": joining dataOps with deep learning recSys with Merlin and Metaflow ([blog](https://medium.com/nvidia-merlin/nvidia-merlin-meets-the-mlops-ecosystem-building-a-production-ready-recsys-pipeline-on-cloud-1a16c156166b))

## Overview

*February 2023*: aside from behavioral testing, the ML pipeline is now completed. A [blog post](https://medium.com/nvidia-merlin/nvidia-merlin-meets-the-mlops-ecosystem-building-a-production-ready-recsys-pipeline-on-cloud-1a16c156166b) on the NVIDIA Medium was just published!

_This_ project is a collaboration with the [Outerbounds](https://outerbounds.com/), [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) and [Comet](https://www.comet.com/signup?utm_source=jacopot&utm_medium=referral&utm_campaign=online_jacopot_2022&utm_content=github_recs_resonable_scale) teams, in an effort to release as open source code a realistic data and ML pipeline for cutting edge recommender systems "that just works". Anyone can ~~[cook](https://medias.spotern.com/spots/w640/192/192480-1554811522.jpg)~~ do great ML, not just Big Tech, if you know how to [pick and choose your tools](https://towardsdatascience.com/tagged/mlops-without-much-ops).

*TL;DR: (after setup) a single ML person is able to train a cutting edge deep learning model (actually, several versions of it in parallel), test it and deploy it without any explicit infrastructure work, without talking to any DevOps person, without using anything that is not Python or SQL.*

As a use case, we pick a popular RecSys challenge, user-item recommendations for the fashion industry: given the past purchases of a shopper, can we train a model to predict what he/she will buy next? In the current V1.0, we target a typical offline training, cached predictons setup: we prepare in advance the top-k recommendations for our users, and store them in a fast cache to be served when shoppers go online.

Our goal is to build a pipeline with all the necessary real-world ingredients:

* dataOps with Snowflake and dbt;
* training Merlin models (possibly on GPUs), in parallel, leveraging Metaflow;
* experiment and parameter tracking;
* advanced testing with Reclist (_FORTHCOMING_);
* serving cached prediction through FaaS and SaaS (AWS Lambda, DynamoDb, the serverless framework);
* error analysis and debugging with a Streamlit app.

At a quick glance, this is what we are building:

![Recs at reasonable scale](/images/stack.jpg)

For an in-depth explanation of the philosophy behind the approach, please check the companion [blog post](https://medium.com/nvidia-merlin/nvidia-merlin-meets-the-mlops-ecosystem-building-a-production-ready-recsys-pipeline-on-cloud-1a16c156166b) or watch our [NVIDIA Summit keynote](https://youtu.be/9rouLchcC0k?t=147).

_If you like this project please add a star on Github here and check out / share / star the [RecList](https://github.com/jacopotagliabue/reclist) package._

### Quick Links

This project builds on our open roadmap for "MLOps at Resonable Scale", automated documentation of pipelines, rounded evaluation for RecSys:

* [NVIDIA Merlin meets the MLOps ecosystem](https://medium.com/nvidia-merlin/nvidia-merlin-meets-the-mlops-ecosystem-building-a-production-ready-recsys-pipeline-on-cloud-1a16c156166b)
* [CIKM RecSys Evaluation Challenge](https://reclist.io/cikm2022-cup/);
* [NVIDIA RecSys Summit keynote](https://youtu.be/9rouLchcC0k?t=147) and [slides](/slides/NVIDIA_RECSYS_SUMMIT_JT.pdf);
* [RecList (project website)](http://reclist.io/);
* [You don't need a bigger boat (repo, paper, talk)](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat).

## Pre-requisites

The code is a self-contained, end-to-end recommender project; however, since we leverage best-in-class tools, some preliminary (one time) setup is required. Please make sure the requirements are satisfied, depending on what you wish to run and on what you are already using - roughly in order of ascending complexity:

_The basics: Metaflow, Snowflake and dbt_

A Snowflake account is needed to host the data, and a working Metaflow setup is needed to run the flow on AWS GPUs if you wish to do so:

* _Snowflake account_: [sign-up for a free trial](https://signup.snowflake.com).
* _AWS account_: [sign-up for a free AWS account](https://aws.amazon.com/free/).
* _Metaflow on AWS_: [follow the setup guide](https://docs.metaflow.org/metaflow-on-aws) - _in theory_ the pipeline should work also with a local setup (i.e. no additional work after installing the `requirements`), if you don't need cloud computing. _However_, we strongly recommend a fully [AWS-compatible setup](https://docs.metaflow.org/metaflow-on-aws). The current flow has been tested with Metaflow out-of-the-box (no config, all local), Metaflow with AWS data store _but_ all local computing, and Metaflow with AWS data store _and_ AWS Batch with GPU computing. 
* _dbt core setup_: on top of installing the package in `requirements.txt`, you need to properly configure your [dbt_profile](https://docs.getdbt.com/dbt-cli/configure-your-profile).

Please note that while the current implementation focused on Metaflow on AWS (with Batch), the exact same code (with a change in decorator!) would work in any [kubernetes-based](https://outerbounds.com/blog/human-centric-data-science-on-kubernetes-with-metaflow/) infrastructure (or [Azure](https://outerbounds.com/blog/metaflow-azure-machine-learning/)!). For the same reasons, Snowflake can be replaced with other warehouses leaving the main result unchanged: an end-to-end, scalable, production-ready pipeline for deep learning recommendations.

_Adding experiment tracking_

* _Comet ML_: [sign-up for free](https://www.comet.com/signup?utm_source=jacopot&utm_medium=referral&utm_campaign=online_jacopot_2022&utm_content=github_recs_resonable_scale) and get an api key. If you don't want experiment tracking, make sure to comment out the Comet specific parts in the `train_model` step.

_Adding PaaS deployment_

* _AWS Lambda setup_: if the env `SAVE_TO_CACHE` is set to `1`, the Metaflow pipeline will try and cache in dynamoDB recommendations for the users in the test set. Those recommendations can be served through an endpont using AWS Lambda. If you wish to serve your recommendations, you need to run the serverless project in the `serverless` folder _before_ running the flow: the project will create _both_ a DynamoDB table and a working GET endpoint. To do so: first, install the [serverless framework](https://www.serverless.com/framework/) and connect it with your [AWS](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/); second, cd into the `serverless` folder, and run `AWS_PROFILE=tooso serverless deploy` (where `AWS_PROFILE` selects a specific AWS config with permission to run the framework, and can be omitted if you use your default). If all goes well, the CLI will create the relevant resources and print out the URL for your public rec API, e.g. `endpoint: GET - https://xafacoa313.execute-api.us-west-2.amazonaws.com/dev/itemRecs`: you can verifiy the endpoint is working by pasting the URL in the browser (response will be empty as you need to run the flow to populate dynamoDB). Make sure the region of deployment in the `serverless.yml` file is the same as the one in the Metaflow pipeline. Note that while we use the _serverless_ framework for convenience, the same setup can be done manually, if preferred.

_Adding a Streamlit app for error analysis_

* We added a small Streamlit app (run with `EXPORT_TO_APP=1` to test this _very experimental feature_) to help visualize and filter predictions: how is the model doing on "short sleeves" items? If you plan on using the app you need to install also the `requirements_app.txt` in the `app` folder.

_A note on containers_

At the moment of writing, Merlin does not have an official ECR, so we pulled `nvcr.io/nvidia/merlin/merlin-tensorflow:22.11` and slightly changed the entry point to work with Metaflow / AWS Batch. The `docker` folder contains the relevant files - the current flow uses a public ECR repository (`public.ecr.aws/outerbounds/merlin-reasonable-scale:22.11-latest`) we prepared on our AWS when running training in _BATCH_; if you wish to use your own ECR or the repo above becomes unavailable for whatever reason, you can just change the relevant `image` parameter in the flow (note: you need to register for a free NVIDIA account first to be able to pull from nvcr).

## Setup

We recommend using python 3.9 for this project.

### Virtual env

Setup a virtual environment with the project dependencies:

* `python -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`

Note that if you never plan on running Merlin's training locally, but only through Metaflow + AWS Batch, you can avoid installing merlin and tensorflow libraries. 

_NOTE_: if you plan on using the Streamlit app (above) make sure to pip install also the `requirements_app.txt` in the `app` folder.

Inside `src`, create a version of the `local.env` file named only `.env` (do _not_ commit it!), and fill its values:

| VARIABLE | TYPE (DEFAULT) | MEANING |
| ------------- | ------------- | ------------- |
| SF_USER | string  | Snowflake user name  |
| SF_PWD | string |  Snowflake password  |
| SF_ACCOUNT | string  |  Snowflake account  |
| SF_DB | string |  Snowflake database  |
| SF_ROLE | string |  Snowflake role to run SQL |
| SF_WAREHOUSE | string |  Snowflake warehouse to run SQL |
| EN_BATCH | 0-1 (0)  | Enable cloud computing for Metaflow |
| COMET_API_KEY | string  | Comet ML api key  |
| EXPORT_TO_APP | 0-1 (0)  | Enable exporting predictions for inspections through Streamlit |
| SAVE_TO_CACHE | 0-1 (0)  | Enable storing predictions to an external cache for serving. If 1, you need to deploy the AWS Lambda (see above) before running the flow  |

### Load data into Snowflake 

The  original dataset is from the [H&M data challenge](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

* Download the files `articles.csv`, `customers.csv`, `transactions_train.csv` and put them in the `src/data` folder. The `data` folder contains `images_to_s3.csv`, which is a simple file simulating a mapping betweend IDs and s3 storage for product images (note that the mapping reflects our own bucket, but you should copy over the images to your own cloud storage and change the files accordingly). Images are *not* used in the RecSys pipeline directly, but they can support additional use cases, such as debugging (which is why we added the meta-data in our sql transformations).
* Run `upload_to_snowflake.py` as a one-off script: the program will dump the dataset to Snowflake, using a typical [modern data stack pattern](https://towardsdatascience.com/the-modern-data-pattern-d34d42216c81). This allows us to use dbt and Metaflow to run a realistic ELT and ML code.

Once you run the script, check your Snowflake for the new tables:

![Raw tables in Snowflake](/images/raw_table.png)

### dbt

After the data is loaded, we use dbt as our transformation tool of choice. While you can run dbt code as part of a Metaflow pipeline, we keep the dbt part separate in this project to simplify the runtime component: it will be trivial (as shown [here](https://github.com/jacopotagliabue/post-modern-stack) for example) to orchestrate the SQL code within Metaflow if you wish to do so. After the data is loaded in Snowflake:

* `cd` into the `dbt` folder;
* run `dbt run`.

Check your Snowflake for the new tables created by dbt:

![Dbt tables](/images/after_dbt.png)

In particular, the table `"EXPLORATION_DB"."HM_POST"."FILTERED_DATAFRAME"` represents a dataframe in which user, article and transaction data are all joined together - the Metaflow pipeline will read from this table, leveraging the pre-processing done at scale through dbt and Snowflake.

## How to run the entire project

### Run the flow

Once the above setup steps are completed, you can run the flow:

* cd into the `src` folder;
* run the flow with `AWS_PROFILE=DemoReno-363597772528 python my_merlin_flow.py run --max-workers 4 --with card`, where `AWS_PROFILE` is needed to select the AWS config that runs the flow and its related AWS infrastructure (you can omit it, if you're using the default). As per standard Metaflow setup, make sure to set as envs also `METAFLOW_PROFILE` and `AWS_DEFAULT_REGION` as needed (you can omit it, if you're using the default settings after the [AWS Setup](https://docs.metaflow.org/getting-started/infrastructure)).

At the end of the flow, you can inspect the default [DAG Card](https://outerbounds.com/blog/integrating-pythonic-visual-reports-into-ml-pipelines/) with `python my_merlin_flow.py card view get_dataset`:

![Metaflow card](/images/card.png)

For an intro to DAG cards, please check our [NeurIPS 2021 paper](https://arxiv.org/abs/2110.13601).

### Results

If you run the flow with the full setup, you will end up with:

* versioned datasets and model artifacts, accessible through the standard [Metaflow client API](https://docs.metaflow.org/metaflow/client);
* a dashboard for [experiment tracking](https://www.comet.com/signup?utm_source=jacopot&utm_medium=referral&utm_campaign=online_jacopot_2022&utm_content=github_recs_resonable_scale), including a quick [panel](/images/predictions.png) to inspect predicted items for selected shoppers;
* an automated, versioned documentation for your pipeline, in the form of Metaflow cards;
* a live, scalable endpoint serving batched predictions using AWS Lambda and DynamoDB.

![Experiment dashboard](/images/tracking.png)

If you have set `EXPORT_TO_APP=1` (and completed the setup), you can also visualize predictions using a Streamlit app that:

* automatically uses the serialized data from the last succesful Metaflow run; 
* leverages CLIP capabilities to offer a quick, free-text way to navigate the prediction set based on the features of the ground truth item (e.g. "long sleeves shirt").

Cd into the `app` folder, and run `streamlit run pred_inspector.py` (make sure Metaflow envs have been set, as usual). You can filter for product type of the target item and use text-to-image search to sort items (try for example with "jeans" or "short sleeves").

![Debugging app](/images/streamlit.gif)

### Where to go from here (either with us or by yourself)?

* Improving error analysis and evaluation: improvements will come automatically from [RecList](https://reclist.io/);
* Making sure dependencies are easy to adjust depending on setup - e.g. dask_cudf vs pandas depending on your GPU set up;
* Supporting other recSys use cases, possibly coming with more complex deployment options (e.g. Triton on Sagemaker).

## Q&A

* _This is a pretty complex project! Can we make it simpler?_ Yes (e.g. [here](https://github.com/jacopotagliabue/post-modern-stack)) and no: while surely the code can be shortened and optimized a bit, in 700 lines we now have _an entire_ recommender system in a production setting, including data preparation in a warehouse, scheduled DAG for batch predictions, a serverless endpoint to serve from the cache, parallel optimization of a modern deep learning architecture (plus, behavioral testing, and CLIP-based data exploration with Streamlit). It is indeed surprising that, when looking into it, the project is really not complex at, all considering you can basically run a production system with it: it is indeed telling that of all the 700 lines, few dozens are actually about the recommendation model, with the majority of the code making sure meta-data, tracking, evaluation etc. are done properly. We are working on other introduction to RecSys, and will add their links here when they are public in case a simplified setup is preferred (e.g. parquet to duckDB instead of setting up the full Snowflake warehouse!).

* _What if my datasets are not static to begin with, but depends on real interactions?_ We open-sourced a [serverless pipeline](https://github.com/jacopotagliabue/paas-data-ingestion) that show how data ingestion could work with the same philosophical principles.

* _I want to add tool X, or replace Y with Z: how modular is this pipeline?_ Our aim is to present a pipeline simple enough to be quickly grasped, complex enough to sustain a real deep learning model and industry use case. That said, it is possible that what worked for us may not work as perfectly for you: e.g. you may wish to change experiment tracking (e.g., an abstraction for [Neptune](https://neptune.ai/) is [here](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat)), or use a different data warehouse solution (e.g. BigQuery), or orchestrate the entire thing in a different way (check again [here](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) for a Prefect-based solution). We start by providing a flow that "just works", but our focus is mainly on the functional pieces, not just the tools: what are the essential computations we need to run a modern recsys pipeline? If you find other tools are better for you, please go ahead - and let us know, feedback is always useful!

## Acknowledgements

Main Contributors:

* [Jacopo](https://www.linkedin.com/in/jacopotagliabue/), general design, Metaflow fan boy, prototype;
* the Outerbounds team, in particular [Hamel](https://www.linkedin.com/in/hamelhusain/) for Metaflow guidance, [Valay](https://www.linkedin.com/in/valay-dave-a3588596/) for AWS Batch support;
* the NVIDIA Merlin team, in particular [Gabriel](https://www.linkedin.com/in/gabrielspmoreira/), [Ronay](https://www.linkedin.com/in/ronay-ak/), [Ben](https://www.linkedin.com/in/ben-frederickson/), [Even](https://www.linkedin.com/in/even-oldridge/).

Special thanks:

* [Dhruv Nair](https://www.linkedin.com/in/dhruvnair/) from [Comet](https://www.comet.com/signup?utm_source=jacopot&utm_medium=referral&utm_campaign=online_jacopot_2022&utm_content=github_recs_resonable_scale) for double-checking our experiment tracking setup and suggesting improvements.

## License

All the code in this repo is freely available under a MIT License, also included in the project.
