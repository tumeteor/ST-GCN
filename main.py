from src.logs import get_logger_settings, setup_logging
import sys
import argparse
import json
import logging
from src.config import Config
from src.data_loader.reader import read_jurbey_from_minio, construct_time_series_traffic_data_using_druid

if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description='Compute Weight for Routing Graph')
    parser.add_argument('--artifact', type=str, help='path to the start2jurbey artifact')
    args = parser.parse_args()
    log_setting = get_logger_settings(logging.INFO)
    setup_logging(log_setting)

    if args.artifact:
        artifact_path = args.artifact
    else:
        artifact_path = cfg.INPUT_PATH

    with open(artifact_path, 'r') as f:
        message = json.load(f)

    logging.info('\u2B07 Getting Jurbey File...')
    g = read_jurbey_from_minio(message['bucket'], message['jurbey_path'])
    logging.info("\u2705 Done loading Jurbey graph.")

    



