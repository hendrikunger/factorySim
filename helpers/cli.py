# cli_args.py
import argparse

def get_args_train():
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

def get_args_inference():
    parser = argparse.ArgumentParser(description="My inference script description")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to the configuration file"
    )

    parser.add_argument(
        "-r", "--rollout",
        action="store_true",
        help="Run rollout using trained policy"
    )
    parser.add_argument(
        "-p","--problemID",
        type=int,
        default=2,
        help="Which - in the list of evaluation environments to use. Default is 1.",
    )

    return parser.parse_args()