from jurbey.jurbey import JURBEY
import tempfile

from src.module import MINIO_CLIENT_GETTER


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


def get_traffic_model_from_minio(bucket_name, interested_date='2019-07-07', interested_hour = '7', prefix='merged_file'):
    """
    Fetch traffic update data from Minio for every hour of a specific day, and the upload them into
    the corresponding S3 bucket. The traffic data can be then consumed for e.g., ETA benchmarking
    Args:
        bucket_name (str): The Minio bucket name where traffic update is store
        interested_date (:obj:`str`, optional): The date that we want for getting the traffic update for all hours
        prefix (:obj:`str`, optional): The prefix of the file name
        s3_path (:obj:`str`, optional): The path that we want to store traffic files in s3

    Returns:
        None
    Raises:
        botocore.exceptions.ClientError: The error occurs when writing to s3
    """
    traffic_objects = MINIO_CLIENT_GETTER().list_objects_v2(bucket_name, prefix=prefix,
                              recursive=False)

    for traffic_obj in traffic_objects:
        if traffic_obj.last_modified.strftime("%Y-%m-%d") == interested_date and traffic_obj.last_modified.hour == interested_hour:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tempf:
                MINIO_CLIENT_GETTER().fget_object(bucket_name, traffic_obj.object_name, tempf.name)




