import argparse
import logging
from utils import load_json

from runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default="./config/ActivityNet.json")
    return parser.parse_args()

def main(args):
    config = load_json(args.config_path)
    print(config)
    runner = Runner(config)
    runner.train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    main(args)
