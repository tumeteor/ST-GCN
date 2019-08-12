import os


class Config:
    def __init__(self):
        self.INPUT_PATH = os.getenv('INPUT_PATH', 'star2jurbey_local.json')
        self.MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio.staging.otonomousmobility.com')
        self.MINIO_PORT = os.getenv('MINIO_PORT', '443')
        self.MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'AKIAIOSFODNN7EXAMPLE')
        self.MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
        self.MINIO_SECURE = os.getenv('MINIO_SECURE', "True")

        self.DB_USERNAME = os.getenv('DB_USERNAME')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_HOSTNAME = os.getenv('DB_HOSTNAME', '127.0.0.1')
        self.DB_NAME = os.getenv('DB_NAME', 'mytaxi')
        self.DB_TABLE_NAME = os.getenv('DB_TABLE_NAME', 'speed_estimates_berlin')
        self.DB_PORT = os.getenv('DB_PORT', '5432')

        self.LDAP_USERNAME = os.getenv('LDAP_USERNAME')
        self.LDAP_PASSWORD = os.getenv('LDAP_PASSWORD')




