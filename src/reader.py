import sys

from jurbey.jurbey import JURBEY
from pandas.io.json import json_normalize
from requests import Session
import tempfile
import pandas as pd
import logging
import json
import networkx as nx

from src.config import Config
from src.graph_utils import partition_graph_by_lonlat
from src.module import MINIO_CLIENT_GETTER, DB_ENGINE_GETTER
cfg = Config()


def read_jurbey_from_minio(bucket_name, object_name):
    from src.module import MINIO_CLIENT

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tempf:
        MINIO_CLIENT.fget_object(bucket_name, object_name, tempf.name)

    with open(tempf.name, 'rb') as tempf:
        g = JURBEY.load(tempf.read())

    return g


def populate_graph_with_max_speed(g):
    for e in g.edges(data=True):
        try:
            g[e[0]][e[1]]["speed"] = float(g[e[0]][e[1]]["data"].metadata.get("maxspeed", 10))
        except ValueError:
            g[e[0]][e[1]]["speed"] = 10
    return g


def _get_traffic_model_from_minio(bucket_name='traffic-update', interested_date='2019-08-02', interested_hour='8',
                                  prefix='merged_file'):
    """
    Fetch traffic update data from Minio for every hour of a specific day, and the upload them into
    the corresponding S3 bucket. The traffic data can be then consumed for e.g., ETA benchmarking
    Args:
        bucket_name (str): The Minio bucket name where traffic update is store
        interested_date (:obj:`str`, optional): The date that we want for getting the traffic update for all hours
        prefix (:obj:`str`, optional): The prefix of the file name
        s3_path (:obj:`str`, optional): The path that we want to store traffic files in s3

    Returns:
        DataFrame
    Raises:
        botocore.exceptions.ClientError: The error occurs when writing to s3
    """
    traffic_objects = MINIO_CLIENT_GETTER().list_objects_v2(bucket_name, prefix=prefix,
                                                            recursive=False)
    df = None
    for traffic_obj in traffic_objects:
        if (traffic_obj.last_modified.strftime("%Y-%m-%d") == interested_date) and \
                (traffic_obj.last_modified.hour == int(interested_hour)):
            print(f"hour: {traffic_obj.last_modified.hour}")
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tempf:
                MINIO_CLIENT_GETTER().fget_object(bucket_name, traffic_obj.object_name, tempf.name)
                df = pd.read_csv(tempf.name, header=0)
                logging.info("Loaded fresh traffic..")
                break
    if df is None:
        logging.warning(f"There is no speed data on date {interested_date} and hour {interested_hour}")
    return df


def _get_traffic_from_postgres(table_name, hour):
    sql = "select from_node as from_osm_id, to_node as to_osm_id, ceil(avg(estimated_speed) * 3.6) as edge_speed_in_km_h " \
          f"from {table_name} as se1 where  indexable_day(timestamp) >= '2019/04/12'::date " \
          f"and indexable_hour(timestamp) = {hour} " \
          "and estimated_speed is not null " \
          "and estimated_speed != 0 " \
          "and estimated_speed < 35 " \
          "group by 1, 2 " \
          "limit 100; "
    store = pd.HDFStore('speed.h5')
    if 'speed' not in store.keys():
        df = pd.read_sql(sql, con=DB_ENGINE_GETTER())
        store.append('speed', df)
    return df


def contruct_time_series_traffic_data(g):
    headers = {
        'Content-type': 'application/json',
    }
    session = Session()
    payload = open('query_segment_timeseries.json', 'r')
    payload = json.load(payload)
    payload['granularity'] = 'hour'
    g = partition_graph_by_lonlat(g)
    dfs = list()
    batch_size = 0
    logging.info(f"Number of edges: {g.number_of_edges()}")
    counter = 0
    for edge in g.edges:
        payload['filter']['fields'][0]['value'] = edge[0]
        payload['filter']['fields'][1]['value'] = edge[1]
        response = session.post(
            url="https://druid.broker.develop.otonomousmobility.com/druid/v2/",
            json=payload,
            auth=(cfg.LDAP_USERNAME, cfg.LDAP_PASSWORD),
            verify=True,
            headers=headers
        )
        if response.status_code != 200:
            raise Exception("Unexpected error:", response.text)
        response = json.loads(response.text)
        df = json_normalize(response)
        df = df['result.average'].transpose()
        df['from_node'] = edge[0]
        df['to_node'] = edge[1]
        dfs.append(df)
        if batch_size == 0:
            with open('dataset.csv', 'w') as f:
                pd.concat(dfs, axis=1).transpose().to_csv(f, header=True)

        if batch_size == 10:
            logging.info(f"at row {counter}")
            counter += 1
            batch_size = 0
            with open('dataset.csv', 'a') as f:
                pd.concat(dfs, axis=0).to_csv(f, header=False)
            dfs = list()

        batch_size += 1


def populate_graph_with_fresh_speed(g):
    fresh_edge_list = list()
    # df = _get_traffic_model_from_minio()
    df = _get_traffic_from_postgres(cfg.DB_TABLE_NAME, hour=7)
    for index, row in df.iterrows():
        try:
            g[int(row[0])][int(row[1])]["fresh_speed"] = float(row[2])
            fresh_edge_list.append((int(row[0]), int(row[1])))
        except KeyError as e:
            pass
            # logging.exception(f"Node does not exist in the graph: {e}")
    return g, fresh_edge_list


def get_dataframe_from_graph(g):
    df = nx.to_pandas_edgelist(g)
    df = df[['source', 'target', 'speed']]
    df.rename(columns={'speed': 'weight'})
    return df

