import tempfile
from jurbey.jurbey import JURBEY
from src.config import Config
cfg = Config()


def read_jurbey_from_minio(bucket_name, object_name):
    from src.module import MINIO_CLIENT

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tempf:
        MINIO_CLIENT.fget_object(bucket_name, object_name, tempf.name)

    with open(tempf.name, 'rb') as tempf:
        g = JURBEY.load(tempf.read())

    return g




