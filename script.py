# script.py
import ray
import os

@ray.remote
def hello_world():
    return "hello world"

# Automatically connect to the running Ray cluster.
runtime_env = {
"env_vars": {"PYTHONWARNINGS": "ignore::UserWarning"},
"working_dir": os.path.join(os.path.dirname(os.path.realpath(__file__))),
"excludes": ["/.git",
            "/.vscode",
            "/wandb",
            "/artifacts",
            "*.skp",
            "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/.git/",
            "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/.vscode/",
            "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/wandb/",
            "/home/sc.uni-leipzig.de/nd67ekek/factorySim/factorySim/artifacts/"],
}




ray.init(runtime_env=runtime_env)

print(ray.get(hello_world.remote()))
