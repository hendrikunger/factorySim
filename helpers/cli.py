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
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Run short tests"
    )
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume a running experiment"
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperparameter optimization"
    )

    return parser.parse_args()