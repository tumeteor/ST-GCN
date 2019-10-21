import os


class Config:
    def __init__(self):
        self.INPUT_PATH = os.getenv('INPUT_PATH', 'star2jurbey_local.json')
        self.MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio.staging.otonomousmobility.com')
        self.MINIO_PORT = os.getenv('MINIO_PORT', '443')
        self.MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
        self.MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
        self.MINIO_SECURE = os.getenv('MINIO_SECURE', "True")





