"""

We run a serverless, "hands-off" stack to train cutting edge deep neural network models for recommendations.

The pipeline is as follows:

    * Snowflake as data warehouse;
    * Metaflow as the pipeline backbone and abstraction over AWS;
    * Merlin for recommendation models;
    * Dynamo (+ lambda) for FaaS/PaaS deployment.

Please check the README and the additional material for the relevant background and context.

"""

from metaflow import FlowSpec, step, batch, Parameter, current, environment
from custom_decorators import enable_decorator, pip
import os
import json
from datetime import datetime


try:
    from dotenv import load_dotenv
    from metaflow_magicdir import magicdir
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    # for remote execution we need to install this before anything else
    # or Python will throw an "undefined decorator" error!
    os.system('pip install metaflow-magicdir==0.0.4')
    from metaflow_magicdir import magicdir
    print("Imported magic folder!")


class myMerlinFlow(FlowSpec):

    ### MERLIN PARAMETERS ###

    MODEL_FOLDER = Parameter(
        name='model_folder',
        help='Folder to store the model from Merlin, between steps',
        default='merlin_model'
    )

    ### DATA PARAMETERS ###

    ROW_SAMPLING = Parameter(
        name='row_sampling',
        help='Snowflake row sampling: if 0, NO sampling is applied',
        default='1'
    )

    #NOTE: data parameters - we split by time, leaving the last two weeks for validation and tests
    # The first date in the table is 2018-09-20
    # The last date in the table is 2020-09-22
    TRAINING_END_DATE = Parameter(
        name='training_end_date',
        help='Data up until this date is used for training, format yyyy-mm-dd',
        default='2020-09-08'
    )

    VALIDATION_END_DATE = Parameter(
        name='validation_end_date',
        help='Data up after training end and until this date is used for validation, format yyyy-mm-dd',
        default='2020-09-15'
    )

    ### TRAINING PARAMETERS ###

    COMET_PROJECT_NAME = Parameter(
        name='comet_project_name',
        help='Name of the project in our Comet dashboard',
        default='two_tower_h_and_m_merlin'
    )

    VALIDATION_METRIC = Parameter(
        name='validation_metric',
        help='Merlin metric to use for picking the best set of hyperparameter',
        default='recall_at_10'
    )

    N_EPOCHS = Parameter(
        name='n_epoch',
        help='Number of epochs to train the Merlin model',
        default='1' # default to 1 for quick testing
    )

    ### SERVING PARAMETERS ###

    DYNAMO_TABLE = Parameter(
        name='dynamo_table',
        help='Name of dynamo db table to store the pre-computed recs. Default is same as in the serverless application',
        default='userItemTable'
    )

    TOP_K = Parameter(
        name='top_k',
        help='Number of products to recommend for a giver shopper',
        default='5'
    )

    @step
    def start(self):
        """
        Start-up: check everything works or fail fast!
        """
        # print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        if os.environ.get('EN_BATCH', '0') == '1':
            print("ATTENTION: AWS BATCH ENABLED!") 
        # we need to check if Metaflow is running with remote (s3) data store or not
        from metaflow.metaflow_config import DATASTORE_SYSROOT_S3 
        print("DATASTORE_SYSROOT_S3: %s" % DATASTORE_SYSROOT_S3)
        if DATASTORE_SYSROOT_S3 is None:
            print("ATTENTION: LOCAL DATASTORE ENABLED")
        # check variables and connections are working fine
        assert os.environ['COMET_API_KEY'] and self.COMET_PROJECT_NAME
        assert int(self.ROW_SAMPLING)
        from snowflake_client import SnowflakeClient
        sf_client = SnowflakeClient(
            os.environ['SF_USER'],
            os.environ['SF_PWD'],
            os.environ['SF_ACCOUNT'],
            os.environ['SF_ROLE'],
            os.environ['SF_WAREHOUSE']
            )
        snowflake_version = sf_client.get_version()
        print(snowflake_version)
        assert snowflake_version is not None
        # check the data range makes sense
        self.training_end_date = datetime.strptime(self.TRAINING_END_DATE, '%Y-%m-%d')
        self.validation_end_date = datetime.strptime(self.VALIDATION_END_DATE, '%Y-%m-%d')
        assert self.validation_end_date > self.training_end_date

        self.next(self.get_dataset)

    @step
    def get_dataset(self):
        """
        Get the data in the right shape from Snowflake, after the dbt transformation
        """
        from snowflake_client import SnowflakeClient
        from pyarrow import Table as pt
        # get SF client
        sf_client = SnowflakeClient(
            os.environ['SF_USER'],
            os.environ['SF_PWD'],
            os.environ['SF_ACCOUNT'],
            os.environ['SF_ROLE'],
            os.environ['SF_WAREHOUSE']
        )
        # check if we need to sample - this is useful to iterate on the code with a real setup
        # without reading in too much data...
        snowflake_sampling = int(self.ROW_SAMPLING)
        sampling_expression = '' if snowflake_sampling == 0 else 'sample({})'.format(snowflake_sampling)
        # thanks to our dbt preparation, the ML models can read in directly the data without additional logic
        query = """
            SELECT 
                ARTICLE_ID,
                PRODUCT_CODE, 
                PRODUCT_TYPE_NO,
                PRODUCT_GROUP_NAME,
                GRAPHICAL_APPEARANCE_NO,
                COLOUR_GROUP_CODE,
                PERCEIVED_COLOUR_VALUE_ID,
                PERCEIVED_COLOUR_MASTER_ID,
                DEPARTMENT_NO,
                INDEX_CODE,
                INDEX_GROUP_NO,
                SECTION_NO,
                GARMENT_GROUP_NO,
                ACTIVE,
                FN,
                AGE,
                CLUB_MEMBER_STATUS,
                CUSTOMER_ID,
                FASHION_NEWS_FREQUENCY,
                POSTAL_CODE,
                PRICE,
                SALES_CHANNEL_ID,
                T_DAT,
                S3_URL
            FROM
                "EXPLORATION_DB"."HM_POST"."FILTERED_DATAFRAME"
                {}
            ORDER BY
                T_DAT ASC
        """.format(sampling_expression)
        print("Fetching rows with query: \n {} \n\nIt may take a while...\n".format(query))
        # fetch raw dataset
        dataset = sf_client.fetch_all(query, debug=True)
        assert dataset
        # convert the classical SNOWFLAKE upper case COLS to lower case (Keras does complain downstream otherwise)
        # TODO: should probably this in Snowflake directly ;-)
        dataset = [{ k.lower(): v for k, v in row.items() } for row in dataset]
        self.item_id_2_meta = { str(r['article_id']): r for r in dataset }
        print("Example articles: {}".format(list(self.item_id_2_meta.keys())[:3]))
        # we split by time window, using the dates specified as parameters
        train_dataset = pt.from_pylist([row for row in dataset if row['t_dat'] < self.training_end_date])
        validation_dataset = pt.from_pylist([row for row in dataset 
            if row['t_dat'] >= self.training_end_date and row['t_dat'] < self.validation_end_date])
        test_dataset = pt.from_pylist([row for row in dataset if row['t_dat'] >= self.validation_end_date])
        print("# {:,} events in the training set, {:,} for validation, {:,} for test".format(
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset)
        ))
        # store and version datasets as a map label -> datasets, for consist processing later on
        self.label_to_dataset = {
            'train': train_dataset,
            'valid': validation_dataset,
            'test': test_dataset
        }
        # go to the next step for NV tabular data
        self.next(self.build_workflow)
    
    # NOTE: we use the magicdir package (https://github.com/outerbounds/metaflow_magicdir)
    # to simplify moving the parquet files that Merlin needs / consumes across steps
    @magicdir(dir='merlin')
    @step
    def build_workflow(self):
        """
        Use NVTabular to transform the original data into the final dataframes for training,
        validation, testing.
        """
        from workflow_builder import get_nvt_workflow, read_to_dataframe # pylint: disable=import-error
        import pandas as pd
        # TODO: find a way to execute dask_cudf when possible and pandas when not
        # import dask as dask, dask_cudf  # pylint: disable=import-error
        import nvtabular as nvt # pylint: disable=import-error
        # read dataset into frames
        label_to_df = {}
        for label, dataset in self.label_to_dataset.items():
            label_to_df[label] = read_to_dataframe(dataset, label)
        full_dataset = nvt.Dataset(pd.concat(list(label_to_df.values())))
        # get the workflow and fit the dataset
        workflow = get_nvt_workflow()
        workflow.fit(full_dataset)
        self.label_to_melin_dataset = {}
        for label, _df in label_to_df.items():
            cnt_dataset = nvt.Dataset(_df)
            self.label_to_melin_dataset[label] = cnt_dataset
            workflow.transform(cnt_dataset).to_parquet(output_path="merlin/{}/".format(label))
        # store the mapping Merlin ID -> article_id and Merlin ID -> customer_id
        user_unique_ids = list(pd.read_parquet('categories/unique.customer_id.parquet')['customer_id'])
        items_unique_ids = list(pd.read_parquet('categories/unique.article_id.parquet')['article_id'])
        self.id_2_user_id = { idx:_ for idx, _ in enumerate(user_unique_ids) }
        self.id_2_item_id = { idx:_ for idx, _ in enumerate(items_unique_ids) }
        # sets of hypers - we serialize them to a string and pass them to the foreach below
        self.hypers_sets = [json.dumps(_) for _ in [
            { 'BATCH_SIZE': 16384 }, #{ 'BATCH_SIZE': 4096 }
        ]]
        self.next(self.train_model, foreach='hypers_sets')

    @environment(vars={
                    'EN_BATCH': os.getenv('EN_BATCH'),
                    'COMET_API_KEY': os.getenv('COMET_API_KEY')
                })
    @enable_decorator(batch(
        #gpu=1, 
        memory=24000,
        image='public.ecr.aws/g2i3l1i3/merlin-reasonable-scale'),
        flag=os.getenv('EN_BATCH'))
    # NOTE: updating requests will just suppress annoying warnings
    @pip(libraries={'requests': '2.28.1', 'comet-ml': '3.26.0'}) 
    @magicdir(dir='merlin')
    @step
    def train_model(self):
        """
        Train models in parallel and store artifacts and validation KPIs for downstream consumption.
        """
        import hashlib
        from comet_ml import Experiment
        import merlin.models.tf as mm
        from merlin.io.dataset import Dataset 
        from merlin.schema.tags import Tags
        import tensorflow as tf
        # this is the CURRENT hyper param JSON in the fan-out
        # each copy of this step in the parallelization will have its own value
        self.hyper_string = self.input
        self.hypers = json.loads(self.hyper_string)
        train = Dataset('merlin/train/*.parquet')
        valid = Dataset('merlin/valid/*.parquet')
        print("Train dataset shape: {}, Validation: {}".format(
            train.to_ddf().compute().shape,
            valid.to_ddf().compute().shape
            ))
        # linking task to experiment
        experiment = Experiment(
            api_key=os.getenv('COMET_API_KEY'), 
            project_name=self.COMET_PROJECT_NAME
            )
        self.comet_experiment_key = experiment.get_key()
        experiment.add_tag(current.pathspec)
        experiment.log_parameters(self.hypers)
        # train the model and evaluate it on validation set
        user_schema = train.schema.select_by_tag(Tags.USER)
        user_inputs = mm.InputBlockV2(user_schema)
        query = mm.Encoder(user_inputs, mm.MLPBlock([128, 64]))
        item_schema = train.schema.select_by_tag(Tags.ITEM)
        item_inputs = mm.InputBlockV2(item_schema,)
        candidate = mm.Encoder(item_inputs, mm.MLPBlock([128, 64]))
        model = mm.TwoTowerModelV2(query, candidate)
        opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        model.compile(
            optimizer=opt, 
            run_eagerly=False, 
            metrics=[mm.RecallAt(10), mm.NDCGAt(10)],)
        model.fit(
            train, 
            validation_data=valid, 
            batch_size=self.hypers['BATCH_SIZE'], 
            epochs=int(self.N_EPOCHS))
        self.metrics = model.evaluate(valid, batch_size=1024, return_dict=True)
        print("\n\n====> Eval results: {}\n\n".format(self.metrics))
        # save the model
        model_hash = str(hashlib.md5(self.hyper_string.encode('utf-8')).hexdigest())
        self.model_path = 'merlin/model{}/'.format(model_hash)
        model.save(self.model_path)
        print("Model saved!")
        self.next(self.join_runs)

    def get_items_topk_recommender_model(
        self,
        train_dataset, 
        model, 
        k: int
    ):
        from merlin.models.utils.dataset import unique_rows_by_features
        from merlin.schema.tags import Tags
        candidate_features = unique_rows_by_features(train_dataset, Tags.ITEM, Tags.ITEM_ID)
        topk_model = model.to_top_k_encoder(candidate_features, k=k, batch_size=128)
        topk_model.compile(run_eagerly=False)

        return topk_model

    @step
    def join_runs(self, inputs):
        """
        Join the parallel runs and merge results into a dictionary.
        """
        # merge results from runs with different parameters (key is hyper settings as a string)
        self.model_paths = { inp.hyper_string: inp.model_path for inp in inputs}
        self.experiment_keys = { inp.hyper_string: inp.comet_experiment_key for inp in inputs}
        self.results_from_runs = { inp.hyper_string: inp.metrics[self.VALIDATION_METRIC] for inp in inputs}
        print("Current results: {}".format(self.results_from_runs))
         # pick one according to some logic, e.g. higher VALIDATION_METRIC
        self.best_model, self_best_result = sorted(self.results_from_runs.items(), key=lambda x: x[1], reverse=True)[0]
        print("Best model is: {}, best path is {}".format(
            self.best_model,
            self.model_paths[self.best_model]
            ))
        # assign the variable for the "final" (the best) model path in S3 and its corresponding name
        self.final_model_path = self.model_paths[self.best_model]
        # pick a final mapping for metadata and other service variables
        self.item_id_2_meta = inputs[0].item_id_2_meta
        self.id_2_item_id = inputs[0].id_2_item_id
        self.id_2_user_id = inputs[0].id_2_user_id
        self.magicdir = inputs[0].magicdir
        # for the best comet experiment, we store the key to upload a json file
        # to explore predictions!
        self.experiment_key = self.experiment_keys[self.best_model]
        # next, for the best model do more testing  
        self.next(self.model_testing)

    def prepare_predictions_for_comet_panel(
        self,
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

    def load_merlin_model(
        self,
        dataset,
        path
    ):
        import tensorflow as tf
        import merlin.models.tf as mm
        loaded_model = tf.keras.models.load_model(path)
        # this is necessary when re-loading the model, before building the top K
        _ = loaded_model(mm.sample_batch(dataset, batch_size=128, include_targets=False))
        # debug
        print("Model re-loaded!")

        return loaded_model

    @environment(vars={'EN_BATCH': os.getenv('EN_BATCH')})
    @enable_decorator(batch(
        #gpu=1, 
        memory=24000,
        image='public.ecr.aws/g2i3l1i3/merlin-reasonable-scale'),
        flag=os.getenv('EN_BATCH'))
    @magicdir(dir='merlin')
    @step
    def model_testing(self):
        """
        Test the generalization abilities of the best model through the held-out set...
        and RecList Beta (Forthcoming!)
        """
        from merlin.io.dataset import Dataset
        import merlin.models.tf as mm
        # loading back datasets and the model for final testing
        train = Dataset('merlin/train/*.parquet')
        test = Dataset('merlin/test/*.parquet')
        loaded_model = self.load_merlin_model(test, self.final_model_path)
        topk_rec_model = self.get_items_topk_recommender_model(test, loaded_model, k=int(self.TOP_K))
        # extract the target item id from the inputs
        test_loader = mm.Loader(test, batch_size=1024, transform=mm.ToTarget(test.schema, "item_id"))
        self.test_metrics = topk_rec_model.evaluate(test_loader, batch_size=1024, return_dict=True)
        print("\n\n====> Test results: {}\n\n".format(self.test_metrics))
        #TODO: add RecList tests!
        # if tests are all good (you could add a flag!) 
        # we can now produce the final list of predictions to be cached 
        # and then served to the shoppers
        self.next(self.saving_predictions)

    @environment(vars={
                    'EN_BATCH': os.getenv('EN_BATCH'),
                    'COMET_API_KEY': os.getenv('COMET_API_KEY')
                })
    @enable_decorator(batch(
        #gpu=1, 
        image='public.ecr.aws/g2i3l1i3/merlin-reasonable-scale'),
        flag=os.getenv('EN_BATCH'))
    @pip(libraries={'requests': '2.28.1', 'comet-ml': '3.26.0'})
    @magicdir(dir='merlin')
    @step
    def saving_predictions(self):
        """
        Run predictions on a target dataset of shoppers (in this case, the testing dataset) and store the predictions
        for the experiment dashboard and the cache downstream.
        """
        from merlin.io.dataset import Dataset
        import merlin.models.tf as mm
        # export ONLY the users in the test set to simulate the set of shoppers we need to recommend items to
        train = Dataset('merlin/train/*.parquet')
        test = Dataset('merlin/test/*.parquet')
        loaded_model = self.load_merlin_model(test, self.final_model_path)
        topk_rec_model = self.get_items_topk_recommender_model(train, loaded_model, k=int(self.TOP_K))
        test_dataset = mm.Loader(test, batch_size=1024, shuffle=False)
        # predict returns a tuple with two elements, scores and product IDs: we get the IDs only
        self.raw_predictions = topk_rec_model.predict(test_dataset)[1]
        n_rows = self.raw_predictions.shape[0]
        self.target_shoppers = test_dataset.data.to_ddf().compute()['customer_id']
        print("Inspect the shopper object for debugging...{}".format(type(self.target_shoppers)))
        # check we have as many predictions as we have shoppers in the test set
        assert n_rows == len(self.target_shoppers)
        # map predictions to a final dictionary, with the actual H and M IDs for users and products
        self.h_m_shoppers = [str(self.id_2_user_id[_]) for _ in self.target_shoppers.to_numpy().tolist()]
        print("Example target shoppers: ", self.h_m_shoppers[:3])
        self.target_items = test_dataset.data.to_ddf().compute()['article_id']
        print("Example target items: ", self.target_items[:3])
        self.best_predictions = self.serialize_predictions(
            self.h_m_shoppers,
            self.id_2_item_id,
            self.raw_predictions,
            self.target_items,
            n_rows
        )
        print("Example target predictions", self.best_predictions[self.h_m_shoppers[0]])
        # log the predictions and close experiment tracking
        self.prepare_predictions_for_comet_panel(
            self.h_m_shoppers,
            self.best_predictions,
            self.item_id_2_meta,
            os.getenv('COMET_API_KEY'),
            self.experiment_key
        )
        # debug, if rows > len(self.predictions), same user appear twice in test set
        print(n_rows, len(self.best_predictions))
        self.next(self.export_to_app)

    def serialize_predictions(
        self,
        h_m_shoppers,
        id_2_item_id,
        raw_predictions,
        target_items,
        n_rows
    ):
        """
        Convert raw predictions to a dictionary user -> items for easy re-use 
        later in the pipeline (e.g. dump the predicted items to a cache!)
        """
        sku_convert = lambda x: [str(id_2_item_id[_]) for _ in x]
        predictions = {}
        for _ in range(n_rows):
            cnt_user = h_m_shoppers[_]
            cnt_raw_preds = raw_predictions[_].tolist()
            cnt_target = target_items[_]
            # don't overwite if we already have a prediction for this user
            if cnt_user not in predictions:
                predictions[cnt_user] = {
                    'items': sku_convert(cnt_raw_preds),
                    'target': sku_convert([cnt_target])[0]
                }

        return predictions

    @step
    def export_to_app(self):
        """
        Prepare artifacts for prediction test cases to be inspected by the app. 

        IMPORTANT: if you wish to run this step, you need the additional dependencies specified
        in the requirements_app.txt file in the app folder. Since these packages are fairly heavy
        and the app is a small prototype, we did not include them in the standard requirements.txt
        and leave it to the user to install them manually if she wishes to explore this feature
        as well.

        If the EXPORT_TO_APP env is not 1, we skip this step and go to deployment.
        """
        if not os.environ.get('EXPORT_TO_APP', None) == '1':
            print("Skipping exporting data to the CLIP-based Streamlit app.")
        else:
            # if the flag is specified, we prepare a dataframe for the app
            import pandas as pd
            from app_utils import encode_image # pylint: disable=import-error
            import torch # pylint: disable=import-error
            from transformers import CLIPProcessor, CLIPModel # pylint: disable=import-error
            # prepare the dataframe containing the predictions, metadata, and images
            rows = []
            # limit number of predictions to a reasonable amount
            max_preds = 100
            for shopper, preds in self.best_predictions.items():
                target_item = preds['target']
                target_img_url = self.item_id_2_meta[target_item]['s3_url']
                top_pred = preds['items'][0]
                predicted_img_url = self.item_id_2_meta[top_pred]['s3_url']
                if not target_img_url or not predicted_img_url:
                    continue
                new_row = {
                    'user_id': shopper,
                    'target_item': target_item,
                    'predicted_item': top_pred,
                    'target_image_url': target_img_url,
                    'predicted_image_url': predicted_img_url,
                    'product_type': self.item_id_2_meta[target_item]['product_group_name']
                }
                rows.append(new_row)
                if len(rows) >= max_preds:
                    break
            # rows to dataframe
            df = pd.DataFrame(rows)
            assert len(df) == max_preds
            # init clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            img_vectors = []
            for img in list(df['target_image_url']):
                cnt_vector = encode_image(
                    model,
                    processor,
                    img,
                    device
                )
                # shape is 1,512 - just save the 1 dim vector
                img_vectors.append(cnt_vector[0])
            df['image_vectors'] = img_vectors
            # save the dataframe as a Metaflow artifact
            self.prediction_df = df

        self.next(self.cache_predictions)

    @step
    def cache_predictions(self):
        """
        Use DynamoDb as a cache and a Lambda (in the serverless folder, check the README)
        to serve pre-computed predictions in a PaaS/FaaS manner.

        Note (see train_model above): we are just storing the predictions for the winning model, as 
        computed in the training step.

        """
        # skip the deployment if not needed
        if not bool(int(os.getenv('SAVE_TO_CACHE'))):
            print("Skipping deployment")
        else:
            print("Caching predictions in DynamoDB")
            import boto3
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(self.DYNAMO_TABLE)
            # upload some static items as a test
            data = [{'userId': user, 'recs': json.dumps(recs) } for user, recs in self.best_predictions.items()] 
            # finally add test user
            data.append({'userId': 'no_user', 'recs': json.dumps(['test_rec_{}'.format(_) for _ in range(int(self.TOP_K))])})
            # loop over predictions and store them in the table
            with table.batch_writer() as writer:
                for item in data:
                    writer.put_item(Item=item)
            print("Predictions are all cached in DynamoDB")

        self.next(self.end)

    @step
    def end(self):
        """
        Just say bye!
        """
        print("All done\n\nSee you, recSys cowboy\n")
        return


if __name__ == '__main__':
    myMerlinFlow()
