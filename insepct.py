import ray, json, pprint, os, time
from ray.util.state import list_actors, list_objects
ray.init(address="auto")



# ------------------------------------------------------------------
# 1) Dump a compact overview of all runners & learners
# ------------------------------------------------------------------
actors = list_actors(detail=True)          # ← returns a list (not a dict)
env_runners = [a for a in actors
               if a["class_name"].startswith("DreamerV3EnvRunner")]
learners     = [a for a in actors
               if "WrappedExecutable" in a["class_name"]]

print(f"{len(env_runners)=}  {len(learners)=}")



pprint.pp([(a["class_name"], a["state"],["pid"])
           for a in env_runners[:3]], compact=True)

# ------------------------------------------------------------------
# 2) Pick the single Learner actor (WrappedExecutable) and query stats
# ------------------------------------------------------------------
if not learners:
    raise SystemExit("❌ no learner actor registered – training never started")

learner= learners[0]

print(learner)
# ray.util.get_actor() works with either name or ID
#learner = ray.get_actor(actor_id=l["actor_id"])

# helper RPCs you patched into sitecustomize earlier
buf_timesteps_1 = ray.get(learner.get_replay_buffer_num_timesteps.remote())
updates_1        = ray.get(learner.get_num_updates.remote())
time.sleep(5)
buf_timesteps_2 = ray.get(learner.get_replay_buffer_num_timesteps.remote())
updates_2        = ray.get(learner.get_num_updates.remote())

print("\nReplay-buffer timesteps:", buf_timesteps_1,
      "→", buf_timesteps_2, f"( Δ {buf_timesteps_2-buf_timesteps_1} in 5 s )")
print("Learner update-calls   :", updates_1,
      "→", updates_2,        f"( Δ {updates_2-updates_1} )")

# ------------------------------------------------------------------
# 3) List big objects still in object-store (optional debug helper)
# ------------------------------------------------------------------
huge = [o for o in list_objects(detail=True)
        if o["object_size"] > 10_000_000]     # >10 MB
print(f"\n{len(huge)} large objects still pinned in the object-store.")
