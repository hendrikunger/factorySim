# cli_args.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="My script description")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "-cid", "--configID", 
        type=int, 
        default=0, 
        help="ID of the configuration in /Experiments/ to use"
    )


    return parser.parse_args()