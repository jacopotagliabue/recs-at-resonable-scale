"""

We run a serverless, "hands-off" stack to train cutting edge deep neural network models for recommendations.

The pipeline is as follows:

    * Snowflake as data warehouse;
    * Metaflow as the pipeline backbone and abstraction over AWS;
    * Merlin for recommendation models;
    * Dynamo (+ lambda) for FaaS/PaaS deployment.

Please check the README and the additional material for the relevant background and context.

"""

from metaflow import FlowSpec, step, batch, S3, Parameter, current, Run, environment
from custom_decorators import enable_decorator, pip
import os
import json
from datetime import datetime


try:
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")


class merlinFlow(FlowSpec):

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
        data_store = os.environ.get('METAFLOW_DATASTORE_SYSROOT_S3', None)
        if data_store is None:
            print("ATTENTION: LOCAL DATASTORE ENABLED")
        # check variables and connections are working fine
        assert os.environ['COMET_API_KEY'] and self.COMET_PROJECT_NAME
        assert int(self.ROW_SAMPLING)
        from snowflake_client import SnowflakeClient
        sf_client = SnowflakeClient(
            os.environ['SF_USER'],
            os.environ['SF_PWD'],
            os.environ['SF_ACCOUNT'],
            os.environ['SF_ROLE']
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
            os.environ['SF_ROLE']
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
                T_DAT
            FROM
                "EXPLORATION_DB"."HM_POST"."FILTERED_DATAFRAME"
                {}
            ORDER BY -- order by date
                T_DAT ASC
        """.format(sampling_expression)
        print("Fetching rows with query: \n {} \n\nIt may take a while...\n".format(query))
        # fetch raw dataset
        dataset = sf_client.fetch_all(query, debug=True)
        assert dataset
        # convert the classical SNOWFLAKE upper case COLS to lower case (Keras does complain downstream otherwise)
        # TODO: should probably this in Snowflake directly ;-)
        dataset = [{ k.lower(): v for k, v in row.items() } for row in dataset]
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
    
    @step
    def build_workflow(self):
        from workflow_builder import get_nvt_workflow, read_to_dataframe # pylint: disable=import-error
        import pandas as pd
        # TODO: find a way to execute dask_cudf when possible and pandas when not
        # import dask as dask, dask_cudf  # pylint: disable=import-error
        import nvtabular as nvt # pylint: disable=import-error
        from dataset_utils import upload_dataset_folders
        # read dataset into frames
        label_to_df = {}
        for label, dataset in self.label_to_dataset.items():
            label_to_df[label] = read_to_dataframe(dataset, label)
        full_dataset = nvt.Dataset(pd.concat(list(label_to_df.values())))
        # get the workflow and fit the dataset
        workflow = get_nvt_workflow()
        workflow.fit(full_dataset)
        for label, _df in label_to_df.items():
            cnt_dataset = nvt.Dataset(_df)
            workflow.transform(cnt_dataset).to_parquet(output_path="{}/".format(label))
        # version the two folders prepared by NV tabular as tar files on s3
        # if the remote datastore is not enabled
        if os.environ.get('METAFLOW_DATASTORE_SYSROOT_S3', None) is not None:
            s3_metaflow_client = S3(run=self)
            self.folders_to_s3_file = upload_dataset_folders(
                s3_client=s3_metaflow_client,
                folders=list(self.label_to_dataset.items())
                )
        # sets of hypers - we serialize them to a string and pass them to the foreach below
        self.hypers_sets = [json.dumps(_) for _ in [
            { 'BATCH_SIZE': 1024 }
        ]]
        self.next(self.train_model, foreach='hypers_sets')

    def get_items_topk_recommender_model(self, train_dataset, schema, model, k):
        from merlin.io.dataset import Dataset 
        from merlin.schema.tags import Tags
        item_features = schema.select_by_tag(Tags.ITEM).column_names
        item_dataset = train_dataset.to_ddf()[item_features].drop_duplicates(subset=['article_id'], keep='last').compute()
        item_dataset = Dataset(item_dataset)
        return model.to_top_k_recommender(item_dataset, k=k)

    @environment(vars={
                    'EN_BATCH': os.getenv('EN_BATCH'),
                    'COMET_API_KEY': os.getenv('COMET_API_KEY')
                })
    @enable_decorator(batch(gpu=1, memory=80000, image='public.ecr.aws/b3x2d2n0/metaflow_merlin'),
                      flag=os.getenv('EN_BATCH'))
    @pip(libraries={'comet-ml': '3.26.0'})
    @step
    def train_model(self):
        """
        Train models in parallel and store KPIs and path for downstream consumption.

        Note: we are now running predictions for all models in parallel over our target set of shoppers (the ones
        in our test set). This is wasteful, as we should run predictions only for the winning model, after we run
        behavioral tests that confirm the model quality - for now, we sidestep the issue of serializing Merlin model
        and restore it by running all predictions and pick downstream the one from the best model.
        """
        from comet_ml import Experiment
        import merlin.models.tf as mm
        from merlin.io.dataset import Dataset 
        import merlin.models.tf.dataset as tf_dataloader
        from merlin.schema.tags import Tags
        from dataset_utils import get_dataset_folders
        # this is the CURRENT hyper param JSON in the fan-out
        # each copy of this step in the parallelization will have its own value
        self.hyper_string = self.input
        self.hypers = json.loads(self.hyper_string)
        target_folder = ''
        # read datasets folder from s3 if datastore is not local
        if os.environ.get('METAFLOW_DATASTORE_SYSROOT_S3', None) is not None:
            s3_metaflow_client = S3(run=self)
            target_folder = 'merlin/' 
            self.local_paths = get_dataset_folders(
                s3_client=s3_metaflow_client,
                folders_to_s3_file=self.folders_to_s3_file,
                target_folder=target_folder
            )
        train = Dataset('{}train/*.parquet'.format(target_folder))
        valid = Dataset('{}valid/*.parquet'.format(target_folder))
        test = Dataset('{}test/*.parquet'.format(target_folder))
        print("Train dataset shape: {}, Validation: {}, Test: {}".format(
            train.to_ddf().compute().shape,
            valid.to_ddf().compute().shape,
            test.to_ddf().compute().shape
            ))
        # linking task to experiment
        experiment = Experiment(
            api_key=os.getenv('COMET_API_KEY'), 
            project_name=self.COMET_PROJECT_NAME
            )
        experiment.add_tag(current.pathspec)
        experiment.log_parameters(self.hypers)
        # train the model
        model = mm.TwoTowerModel(
            train.schema,
            query_tower=mm.MLPBlock([128, 64], no_activation_last_layer=True),
            samplers=[mm.InBatchSampler()],
            embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True)
        )
        model.compile(optimizer="adam", run_eagerly=False, metrics=[mm.RecallAt(10), mm.NDCGAt(10)])
        model.fit(train, validation_data=valid, batch_size=self.hypers['BATCH_SIZE'], epochs=int(self.N_EPOCHS))
        # test the model on validation set and store the results in a MF variable
        self.metrics = model.evaluate(valid, batch_size=1024, return_dict=True)
        print("\n\n====> Eval results: {}\n\n".format(self.metrics))
        experiment.end()
        # export ONLY the users in the test set to simulate the set of shoppers we need to recommend items to
        # first, we provide train set as a corpus
        topk_rec_model = self.get_items_topk_recommender_model(
            train, train.schema, model, k=int(self.TOP_K)
        )
        test_dataset = tf_dataloader.BatchedDataset(
            test, batch_size=1024, shuffle=False,
        )
        # then, we predict on the test set
        self.predictions = topk_rec_model.predict(test_dataset)[1]
        print(self.predictions.shape)
        # TODO: join predictions with item IDs and user IDs
        # TODO: decide how to best serialize the model in a MF variable
        self.model_path = '' # upload model to s3
        self.next(self.join_runs)

    @step
    def join_runs(self, inputs):
        """
        Join the parallel runs and merge results into a dictionary.
        """
        # merge results from runs with different parameters (key is hyper settings as a string)
        # and collect the predictions made by the different versions
        self.model_paths = { inp.hyper_string: inp.model_path for inp in inputs}
        self.results_from_runs = { inp.hyper_string: inp.metrics[self.VALIDATION_METRIC] for inp in inputs}
        self.all_predictions = { inp.hyper_string: inp.predictions for inp in inputs}
        print("Current results: {}".format(self.results_from_runs))
         # pick one according to some logic, e.g. higher VALIDATION_METRIC
        self.best_model, self_best_result = sorted(self.results_from_runs.items(), key=lambda x: x[1], reverse=True)[0]
        self.best_predictions = self.all_predictions[self.best_model]
        print("Best model is: {}, path is {}".format(
            self.best_model,
            self.model_paths[self.best_model]
            ))
        # next, deploy
        self.next(self.model_testing)

    @step
    def model_testing(self):
        """
        Test the generalization abilities of the best model through RecList

        Forthcoming!
        """
        #TODO: add RecList tests, for now just go the last step
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
            data = []
            # test user first
            data.append({ 'userId': 'no_user', 'recs': json.dumps(['test_rec_{}'.format(_) for _ in range(3)])})
            # loop over predictions and store them
            for idx in range(len(self.best_predictions)):
                data.append({ 'userId': f'user_{idx}', 'recs': json.dumps(self.best_predictions[idx].tolist()) })
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
    merlinFlow()
