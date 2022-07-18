import json
import os
import time
import boto3
import uuid


# read table name from env and check for existence
DYNAMO_TABLE_NAME = os.environ.get('TABLE_NAME', None)
assert DYNAMO_TABLE_NAME is not None
# init the client, make sure the region is same as in yml file!
dynamodb = boto3.resource('dynamodb', region_name="us-west-2")
table = dynamodb.Table(DYNAMO_TABLE_NAME)


def wrap_response(status_code: int, items: list):
    response = {
        'data': items,
        'metaData': {
            'statusCode': status_code,
            'eventId': str(uuid.uuid4()),
            'timestamp': time.time()
        }
    }
    # print in cloudwatch for debug!
    print(response)
    return {
        'statusCode': status_code,
        'headers': {
            # this makes the function callable across domains!
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json' 
        },
        'body': json.dumps(response)
        }


def user_to_item_recs(event, context):
    """
    Retrieve pre-computed recs for a target user.

    Example GET query: https://xxx.execute-api.us-east-1.amazonaws.com/dev/itemRecs?userId=my_user_id
    where xxx is the url unique to your lambda function, and my_user_id is the user id 
    you want to get recs for.
    """
    # variables we are going to return to the client
    status_code = 200
    items = []
    # try/except is used for error handling
    try:
        params = event['queryStringParameters']
        # TODO: if user Id is not provided, we should return an error
        userId = params.get('userId', 'no_user') if params is not None else 'no_user'
        # debug
        print('Current userId: {}'.format(userId))
        response = table.get_item(
            Key={'userId': userId }
        )
        # debug
        print(response)
        # check if we have any recs available for the user
        if 'Item' in response:
            # if we have, return them after reading the string into a list
            items = json.loads(response['Item']['recs'])

    except Exception as err:
        # just print it out in Cloudwatch
        print(str(err))
        status_code = 400
        items = []
    
    # return the response to the client
    return wrap_response(status_code, items)


