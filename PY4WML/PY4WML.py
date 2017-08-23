import json

import pandas as pd
import requests
import urllib3
from repository.mlrepositoryartifact import MLRepositoryArtifact
from repository.mlrepositoryclient import MLRepositoryClient


def save_model(model, name, training_data, watson_ml_creds=None):
    ml_repository_client = MLRepositoryClient() if watson_ml_creds is None else MLRepositoryClient(
        watson_ml_creds['url'])
    if watson_ml_creds is not None:
        ml_repository_client.authorize(watson_ml_creds['username'], watson_ml_creds['password'])
    # print vars(ml_repository_client.api_client)
    saved_model = ml_repository_client.models.save(MLRepositoryArtifact(model, name=name, training_data=training_data))
    return {'saved_model': saved_model, 'loaded_artifact': ml_repository_client.models.get(saved_model.uid)}


def print_saved_info(saved):
    saved_model = saved['saved_model']
    loaded_artifact = saved['loaded_artifact']
    print "modelType:", saved_model.meta.prop("modelType")
    print "trainingDataSchema:", str(saved_model.meta.prop("trainingDataSchema"))
    print "creationTime:", str(saved_model.meta.prop("creationTime"))
    print "modelVersionHref:", saved_model.meta.prop("modelVersionHref")
    print "label:", saved_model.meta.prop("label")
    print "uid:", str(saved_model.uid)
    print "name:", loaded_artifact.name


def get_ml_token(watson_ml_creds):
    if 'ml_token' not in watson_ml_creds:
        response = requests.get(
            watson_ml_creds['url'] + '/v3/identity/token',
            headers=urllib3.util.make_headers(
                basic_auth=':'.join([watson_ml_creds['username'], watson_ml_creds['password']])))
        watson_ml_creds['ml_token'] = json.loads(response.text).get('token')
    return watson_ml_creds['ml_token']


def get_headers(watson_ml_creds):
    return {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + get_ml_token(watson_ml_creds)}


def get_instance_info(watson_ml_creds):
    if 'instance_info' not in watson_ml_creds:
        response_get_instance = requests.get(
            watson_ml_creds['url'] + "/v3/wml_instances/" + watson_ml_creds['instance_id'],
            headers=get_headers(watson_ml_creds))
        watson_ml_creds['instance_info'] = json.loads(response_get_instance.text)
    return watson_ml_creds['instance_info']


def get_models_info(watson_ml_creds):
    response_get = requests.get(
        get_instance_info(watson_ml_creds).get('entity').get('published_models').get('url'),
        headers=get_headers(watson_ml_creds))
    return json.loads(response_get.text)


def get_deployment_url(models_info, uid):
    [endpoint_deployments] = [x.get('entity').get('deployments').get('url') for x in models_info.get('resources') if
                              x.get('metadata').get('guid') == uid]
    return endpoint_deployments


def create_scoring_url(watson_ml_creds, deployment_url, name, description='notebook deployed model'):
    response_online = requests.post(
        deployment_url, json={'name': name, 'description': description, 'type': 'online'},
        headers=get_headers(watson_ml_creds))
    return json.loads(response_online.text).get('entity').get('scoring_url')


def score(watson_ml_creds, scoring_url, payload):
    response_scoring = requests.post(
        scoring_url, json=payload,
        headers=get_headers(watson_ml_creds))
    return json.loads(response_scoring.text)


def scores_to_dataframe(scores, columns=None):
    columns_idx = []
    results_dict = {}
    for i in range(0, len(scores['fields'])):
        if columns is None or scores['fields'][i] in columns:
            results_dict[scores['fields'][i]] = []
            columns_idx.append(i)
    for rec in scores['values']:
        for i in columns_idx:
            results_dict[scores['fields'][i]].append(rec[i])
    return pd.DataFrame.from_dict(results_dict)
