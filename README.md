# recs-at-resonable-scale
Recommendations at "Reasonable Scale": joining dataOps with deep learning recSys with Merlin and Metaflow

## Overview

*July 2022*: this is a WIP, come back often for updates, a blog post and my NVIDIA talk (FORTHCOMING)!

_This_ project is a collaboration with the [Outerbounds](https://outerbounds.com/) and [NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin) teams, in an effort to release as open source code a realistic data and ML pipeline for cutting edge recommender systems "that just works". Anyone can ~~[cook](https://medias.spotern.com/spots/w640/192/192480-1554811522.jpg)~~ do great ML, not just Big Tech, if you know how to [pick and choose your tools](https://towardsdatascience.com/tagged/mlops-without-much-ops).

*TL;DR: (after setup) a single ML person is able to train a cutting edge deep learning model (actually, several versions of it in parallel), test it and deploy it without any explicit infrastructure work, without talking to any DevOps person, without using anything that is not Python or SQL.*

As a use case, we pick a popular RecSys challenge, user-item recommendations for the fashion industry: given the past purchases of a shopper, can we train a model to predict what he/she will buy next? In the current V1.0, we target a typical offline training, cached predictons setup: we prepare in advance the top-k recommendations for our users, and store them in a fast cache to be served when shoppers go online.

Our goal is to build a pipeline with all the necessary real-world ingredients:

* dataOps with Snowflake and dbt;
* training Merlin models on GPUs, in parallel, leveraging Metaflow;
* advanced testing with Reclist (_FORTHCOMING_);
* serving cached prediction through FaaS and SaaS (AWS Lambda, DynamoDb, the serverless framework).

At a quick glance, this is what we are building:

![Recs at reasonable scale](/images/stack.jpg)

For an in-depth explanation of the philosophy behind the approach, please check the companion blog post (FORTHCOMING).

### Quick Links

This project builds on our open roadmap for "MLOps at Resonable Scale", automated documentation of pipelines, rounded evaluation for RecSys:

* _NEW_: Upcoming [CIKM RecSys Evaluation Challenge](https://reclist.io/cikm2022-cup/);
* [RecList (project website)](http://reclist.io/);
* [You don't need a bigger boat (repo, paper, talk)](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat);
* [Post-Modern Stack (repo)](https://github.com/jacopotagliabue/post-modern-stack);
* [DAG Cards are the new model cards (NeurIPS paper)](https://arxiv.org/abs/2110.13601).

## Pre-requisites

The code is a self-contained, end-to-end recommender project; however, since we leverage best-in-class tools, some preliminary (one time) setup is required. Please make sure the requirements are satisfied, depending on what you wish to run and on what you are already using - roughly in order of ascending complexity:

_The basics: Metaflow, Snowflake and dbt_

A Snowflake account is needed to host the data, and a working Metaflow setup is needed to run the flow on AWS GPUs if you wish to do so:

* _Snowflake account_: [sign-up for a free trial](https://signup.snowflake.com).
* _AWS account_: [sign-up for a free AWS account](https://aws.amazon.com/free/).
* _Metaflow on AWS_: [follow the setup guide](https://docs.metaflow.org/metaflow-on-aws) - _in theory_ the pipeline should work also with a local setup (i.e. no additional work after installing the `requirements`), if you don't need cloud computing. _However_, we strongly recommend a fully [AWS-compatible setup](https://docs.metaflow.org/metaflow-on-aws); 
* _dbt core setup_: on top of installing the package in `requirements.txt`, you need to properly configure your [dbt_profile](https://docs.getdbt.com/dbt-cli/configure-your-profile).

_Adding experiment tracking_

* _Comet ML_: [sign-up for free](https://www.comet.ml/signup) and get an api key. If you don't want experiment tracking, make sure to comment out the Comet specific parts in the `train_model` step.

_Adding PaaS deployment_

* _AWS Lambda setup_: if the env `SAVE_TO_CACHE` is set to `1`, the Metaflow pipeline will try and cache in dynamoDB recommendations for the users in the test set. Those recommendations can be served through an endpont using AWS Lambda. If you wish to serve your recommendations, you need to run the serverless project in the `serverless` folder _before_ running the flow: the project will create _both_ a DynamoDB table and a working GET endpoint. To do so: first, install the [serverless framework](https://www.serverless.com/framework/) and connect it with your [AWS](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/); second, cd into the `serverless` folder, and run `AWS_PROFILE=tooso serverless deploy` (where `AWS_PROFILE` selects a specific AWS config with permission to run the framework, and can be omitted if you use your default). If all goes well, the CLI will create the relevant resources and print out the URL for your public rec API, e.g. `endpoint: GET - https://xafacoa313.execute-api.us-west-2.amazonaws.com/dev/itemRecs`: you can verifiy the endpoint is working by pasting the URL in the browser (response will be empty as you need to run the flow to populate dynamoDB). Make sure the region of deployment in the `serverless.yml` file is the same as the one in the Metaflow pipeline. Note that while we use the _serverless_ framework for convenience, the same setup can be done manually, if preferred.

_A note on containers_

At the moment of writing, Merlin does not have an official ECR, so we pulled the following images:

* `nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.05`

and slightly changed the entry point to work with Metaflow. The `docker` folder contains the relevant files - the current flow uses a public ECR repository we prepared on our AWS (`public.ecr.aws/b3x2d2n0/metaflow_merlin`) when running training in _BATCH_; if you wish to use your own ECR or the repo above becomes unavailable for whatever reason, you can just change the relevant `image` parameter in the flow.

## Setup

We recommend using python 3.8 for this project.

### Virtual env

Setup a virtual environment with the project dependencies:

* `python -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`

Note that if you never plan on running Merlin's training locally, but only through AWS Batch, you can avoid installing merlin and tensorflow libraries to run the flow. 

Inside `src`, create a version of the `local.env` file named only `.env` (do _not_ commit it!), and fill its values:

| VARIABLE | TYPE (DEFAULT) | MEANING |
| ------------- | ------------- | ------------- |
| SF_USER | string  | Snowflake user name  |
| SF_PWD | string |  Snowflake password  |
| SF_ACCOUNT | string  |  Snowflake account  |
| SF_DB | string |  Snowflake database  |
| SF_ROLE | string |  Snowflake role to run SQL |
| EN_BATCH | 0-1 (0)  | Enable cloud computing for Metaflow |
| COMET_API_KEY | string  | Comet ML api key  |
| SAVE_TO_CACHE | 0-1 (0)  | Enable storing predictions to an external cache for serving. If 1, you need to deploy the AWS Lambda (see above) before running the flow  |

### Load data into Snowflake 

The  original dataset is from the [H&M data challenge](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

* Download the files `articles.csv`, `customers.csv`, `transactions_train.csv` and put them in the `src/data` folder.
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
* run the flow with `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_merlin_flow.py --package-suffixes ".py" run –-with card --max-workers 4`, where `METAFLOW_PROFILE` is needed to select a specific Metaflow config (you can omit it, if you're using the default), `AWS_PROFILE` is needed to select a specific AWS config that runs the flow and it's related AWS infrastructure (you can omit it, if you're using the default), and `AWS_DEFAULT_REGION` is needed to specify the target AWS region (you can omit it, if you've it already specified in your local AWS PROFILE and you do not wish to change it).

At the end of the flow, you can inspect the default [DAG Card](https://outerbounds.com/blog/integrating-pythonic-visual-reports-into-ml-pipelines/) with `METAFLOW_PROFILE=metaflow AWS_PROFILE=tooso AWS_DEFAULT_REGION=us-west-2 python my_merlin_flow.py card view get_dataset`:

![Metaflow card](/images/card.png)

### Results

If you run the flow with the full setup, you will end up with:

* versioned datasets and model artifacts, accessible through the standard [Metaflow client API](https://docs.metaflow.org/metaflow/client);
* a dashboard for experiment tracking;
* an automated, versioned documentation for your pipeline, in the form of Metaflow cards;
* a live, scalable endpoint serving batched predictions using AWS Lambda and DynamoDB.

### TODOs

* we are now running predictions for all models in parallel over our target set of shoppers. This is wasteful, as we should run predictions only for the winning model, after we run tests that confirm model quality - for now, we sidestep the issue of serializing Merlin model and restore it;
* make sure dependencies are easy to adjust depending on setup - e.g. dask_cudf vs pandas depending on your set up;
* support other recSys use cases, possibly coming with more complex deployment options (e.g. Triton on Sagemaker);
* improve testing with RecList, when [RecList Beta](https://reclist.io/) is ready.

## What's next?

_TBC_

## Q&A

* _What if my datasets are not static to begin with, but depends on real interactions?_ We open-sourced a [serverless pipeline](https://github.com/jacopotagliabue/paas-data-ingestion) that show how data ingestion could work with the same philosophical principles.

* _I want to add tool X, or replace Y with Z: how modular is this pipeline?_ Our aim is to present a pipeline simple enough to be quickly grasped, complex enough to sustain a real deep learning model and industry use case. That said, it is possible that what worked for us may not work as perfectly for you: e.g. you may wish to change experiment tracking (e.g., an abstraction for [Neptune](https://neptune.ai/) is [here](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat)), or use a different data warehouse solution (e.g. BigQuery), or orchestrate the entire thing in a different way (check again [here](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) for a Prefect-based solution). We start by providing a flow that "just works", but our focus is mainly on the functional pieces, not just the tools: what are the essential computations we need to run a modern recsys pipeline? If you find other tools are better for you, please go ahead - and let us know, feedback is always useful!

_TBC_

## Acknowledgements

Main Contributors:

* [Jacopo](https://www.linkedin.com/in/jacopotagliabue/), general design, Metaflow fan boy, prototype;
* the Outerbounds team, in particular [Hamel](https://www.linkedin.com/in/hamelhusain/) for Metaflow guidance, [Valay](https://www.linkedin.com/in/valay-dave-a3588596/) for AWS Batch support;
* the NVIDIA Merlin team, in particular [Gabriel](https://www.linkedin.com/in/gabrielspmoreira/), [Ronay](https://www.linkedin.com/in/ronay-ak/), [Ben](https://www.linkedin.com/in/ben-frederickson/), [Even](https://www.linkedin.com/in/even-oldridge/).

Special thanks:

* [Dhruv Nair](https://www.linkedin.com/in/dhruvnair/) for double-checking our experiment tracking setup and suggesting improvements.

## License

All the code in this repo is freely available under a MIT License, also included in the project.
