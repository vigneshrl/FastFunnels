# Issue 1
``` bash
  File "/u/atk9sb/.local/lib/python3.12/site-packages/marshmallow/__init__.py", line 4, in <module>
    from marshmallow.schema import (
  File "/u/atk9sb/.local/lib/python3.12/site-packages/marshmallow/schema.py", line 5, in <module>
    from collections import defaultdict, Mapping, namedtuple
ImportError: cannot import name 'Mapping' from 'collections' (/sw/ubuntu2204/ebu082025/software/common/core/miniforge/25.3.1-py3.12/lib/python3.12/collections/__init__.py)
```
This is happening beacuse "The root cause is that marshmallow 2.15.3 tries to import Mapping from collections, but in Python 3.10+, it was moved to collections.abc. Upgrading marshmallow will fix this."

1. Load your conda environment and upgrade your conda environment
``` bash 
pip install --upgrade marshmallow
```

# Issue 2 
``` bash 
 File "/bigtemp/atk9sb/FastFunnels/scripts/training.py", line 59, in _init
    env.reset(seed=seed + rank)
  File "/bigtemp/atk9sb/FastFunnels/scripts/patch_sempc.py", line 1470, in reset
    self.base_env = gym.make(
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/site-packages/gymnasium/envs/registration.py", line 741, in make
    env_spec = _find_spec(id)
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/site-packages/gymnasium/envs/registration.py", line 527, in _find_spec
    _check_version_exists(ns, name, version)
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/site-packages/gymnasium/envs/registration.py", line 393, in _check_version_exists
    _check_name_exists(ns, name)
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/site-packages/gymnasium/envs/registration.py", line 370, in _check_name_exists
    raise error.NameNotFound(
gymnasium.error.NameNotFound: Environment `f1tenth` doesn't exist.
Traceback (most recent call last):
  File "/bigtemp/atk9sb/FastFunnels/scripts/training.py", line 458, in <module>
    train_patch_policy(
  File "/bigtemp/atk9sb/FastFunnels/scripts/training.py", line 219, in train_patch_policy
    env = SubprocVecEnv([
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/site-packages/stable_baselines3/common/vec_env/subproc_vec_env.py", line 127, in __init__
    observation_space, action_space = self.remotes[0].recv()
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "/u/atk9sb/.conda/envs/f1tenth_gym/lib/python3.9/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer
```
You might see this error as sometimes the editable install might be broken, try the following:

1. Try to import the f1tenth_gym after you have created the base_en. Since you are using the SubProcVecEnv each environment runs in a seperate child process. The import f1tenth_gym inside the __init() should handle this. But the issue is that gym.make("f1tenth_gym:f1tenth-v0") relies on the gymnasium entry point registration, which may not work if f1tenth_gym was installed in development/editable mode incorrectly.

This should make it work, The root cause:

>[!NOTE] 
> 1. SubprocVecEnv spawns separate Python processes for each environment
> 2. Each subprocess needs f1tenth_gym imported to register the f1tenth-v0 environment with gymnasium
> 3. Your import f1tenth_gym in make_patch_env._init() (training.py:43) runs before PatchEnv() is created, but the environment registration happens via entry points — if f1tenth_gym isn't installed properly via pip install -e ., the entry point may not auto-register
> 4. Adding the explicit import right before gym.make() in patch_sempc.py ensures registration happens in the subprocess regardless of how it was installed
>