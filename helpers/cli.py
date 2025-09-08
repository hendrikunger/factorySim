# cli_args.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="My script description")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True,
        default="config.yaml", 
        help="Path to the configuration file"
    )
    


    return parser.parse_args()