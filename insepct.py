import ray, json, pprint
ray.init(address="auto")
from ray._private.state import actors
pprint.pp([ (a['ActorClassName'], a['State']) for a in actors().values()
            if a['ActorClassName'].startswith(('DreamerV3EnvRunner',
                                                'create_executable_class')) ])