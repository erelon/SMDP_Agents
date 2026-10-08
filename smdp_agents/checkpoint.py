"""Save and load any agent in this library — tabular or network.

    agent.save("run.pkl")
    agent = smdp_agents.checkpoint.load("run.pkl")

Unlike :mod:`smdp_agents.portable_policy`, which imports a POLICY produced by a
different training stack, this round-trips a whole agent: its table or network,
its rate estimator, its optimiser, its RNG and its step counters. An agent
restored from a checkpoint continues exactly where it stopped.

WHAT IS CAPTURED, and why it is phrased as an exclusion. Everything in the
agent's ``__dict__`` is saved except a short, named list of things that are not
state (a live environment reference) or that need special handling (the torch
modules). A whitelist would be easier to read and would silently drop any
attribute added later — and a checkpoint that quietly forgets the rate estimate
is worse than no checkpoint at all, because the agent still runs and merely
starts from the wrong place. So anything new is captured by default, and
anything that cannot be pickled fails loudly at save time.

Torch tensors are converted to numpy on the way out and back on the way in, so a
checkpoint does not carry a pickled torch object and can be read by a different
torch version.

The file is a pickle: load checkpoints you produced or trust, as with any
checkpoint format.
"""
import importlib
import pickle

import numpy as np

#: Not state: a live environment reference belongs to the caller, not the agent.
SKIP = ("env",)
#: Handled explicitly rather than pickled, because they are torch objects.
TORCH_ATTRS = ("net", "optimizer")

FORMAT = 1


def _to_numpy(obj):
    """Recursively replace torch tensors with numpy arrays."""
    import torch
    if torch.is_tensor(obj):
        return {"__tensor__": obj.detach().cpu().numpy()}
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_to_numpy(v) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


def _from_numpy(obj):
    """The inverse of :func:`_to_numpy`."""
    import torch
    if isinstance(obj, dict):
        if set(obj) == {"__tensor__"}:
            return torch.as_tensor(obj["__tensor__"])
        return {k: _from_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_from_numpy(v) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


def state_of(agent):
    """The complete restorable state of ``agent`` as plain python + numpy."""
    import torch
    attrs = {}
    for key, value in vars(agent).items():
        if key in SKIP or key in TORCH_ATTRS:
            continue
        try:
            pickle.dumps(value)
        except Exception as exc:                      # noqa: BLE001
            raise TypeError(
                "agent attribute %r cannot be checkpointed (%s). Add it to "
                "SKIP if it is not state, or give it a picklable form."
                % (key, exc))
        attrs[key] = value

    blob = {"format": FORMAT,
            "class": "%s.%s" % (type(agent).__module__, type(agent).__qualname__),
            "attrs": attrs}

    if hasattr(agent, "net"):
        blob["net"] = {k: v.detach().cpu().numpy()
                       for k, v in agent.net.state_dict().items()}
        blob["arch"] = {"obs_dim": agent.obs_dim, "act_dim": agent.act_dim,
                        "hidden": tuple(agent.hidden),
                        "init_log_std": agent.init_log_std,
                        "discrete": bool(agent.discrete)}
        if getattr(agent, "optimizer", None) is not None:
            blob["optimizer"] = _to_numpy(agent.optimizer.state_dict())
        # torch's RNG is GLOBAL, not the agent's, but a network agent draws its
        # actions and its minibatch order from it -- so an exact resume needs it.
        # Saved always, restored only on request: loading an object should not
        # silently reseed the interpreter.
        blob["torch_rng"] = torch.get_rng_state().numpy()
    return blob


def save(agent, path):
    """Write ``agent``'s complete state to ``path``."""
    with open(path, "wb") as fh:
        pickle.dump(state_of(agent), fh, protocol=4)
    return path


def load(path, device=None, restore_torch_rng=False):
    """Rebuild the agent written by :func:`save`.

    ``restore_torch_rng`` also restores torch's GLOBAL random state, which a
    network agent uses for action sampling and minibatch order. Without it a
    restored agent is identical but its random draws continue from wherever the
    process happens to be, so a resumed run is statistically equivalent rather
    than step-for-step identical. It is off by default because loading an object
    should not reseed the interpreter behind the caller's back.
    """
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if blob.get("format") != FORMAT:
        raise ValueError("unknown checkpoint format %r" % blob.get("format"))

    module_name, _, class_name = blob["class"].rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)

    # Bypass __init__: the saved __dict__ IS the constructed state, and replaying
    # the constructor would re-seed the RNG and rebuild the estimators.
    agent = cls.__new__(cls)
    agent.__dict__.update(blob["attrs"])
    agent.env = None

    if "net" in blob:
        import torch
        from .gaussian_mlp import CategoricalMLP, GaussianMLP
        arch = blob["arch"]
        dev = device or agent.__dict__.get("device", "cpu")
        if arch["discrete"]:
            net = CategoricalMLP(arch["obs_dim"], arch["act_dim"], arch["hidden"])
        else:
            net = GaussianMLP(arch["obs_dim"], arch["act_dim"], arch["hidden"],
                              arch["init_log_std"])
        net.load_state_dict({k: torch.as_tensor(v) for k, v in blob["net"].items()},
                            strict=True)
        agent.net = net.to(dev)
        agent.device = dev
        agent.optimizer = None
        if "optimizer" in blob:
            from torch import optim
            opt_state = _from_numpy(blob["optimizer"])
            # lr comes from the SAVED optimiser, not from the agent: plain PPO
            # does not keep a learning_rate attribute (only the tabular bases do).
            lr = opt_state["param_groups"][0]["lr"]
            agent.optimizer = optim.Adam(agent.net.parameters(), lr=lr, foreach=True)
            agent.optimizer.load_state_dict(opt_state)

    if restore_torch_rng and "torch_rng" in blob:
        import torch
        torch.set_rng_state(torch.as_tensor(blob["torch_rng"], dtype=torch.uint8))
    return agent
