import os
from pathlib import Path
import numpy as np

# Heavy imports deferred inside Predictor to avoid loading if not needed
class QCNetPredictor:
    def __init__(self, ckpt_path, device=None):
        if device is None:
            device = "cuda" if (hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available()) else "cpu"
        self.device = device
        # Lazy import torch and lightning
        import torch
        from QCNet.model.qcnet import QCNet   # adjust import per actual repo
        print(f"[QCNetWrapper] Loading checkpoint {ckpt_path} on {device}")
        self.model = QCNet.load_from_checkpoint(ckpt_path, map_location=device)
        self.model.eval().to(device)

    def _world_to_heterodata(self, world):
        """Convert CARLA world snapshot to QCNet HeteroData.

        Placeholder – implement with actual sensor/actor extraction.
        """
        raise NotImplementedError

    def __call__(self, world):
        import torch
        data = self._world_to_heterodata(world)
        data = data.to(self.device)
        with torch.no_grad():
            output = self.model(data)
        traj = output['traj'].cpu().numpy()  # (N,K,T,2)
        probs = output['prob'].cpu().numpy()  # (N,K)
        return traj, probs

_predictor = None

def predict(world, num_agents=10, k=6, horizon=12):

    global _predictor
    ckpt = os.environ.get("QCNET_CKPT", "")
    if not ckpt or not Path(ckpt).exists():
        return (np.zeros((num_agents, k, horizon, 2), dtype=np.float32),
                np.full((num_agents, k), 1.0 / k, dtype=np.float32))
    if _predictor is None:
        _predictor = QCNetPredictor(ckpt)
    try:
        return _predictor(world)
    except NotImplementedError:
        print("[QCNetWrapper] _world_to_heterodata not implemented – returning zeros.")
        return (np.zeros((num_agents, k, horizon, 2), dtype=np.float32),
                np.full((num_agents, k), 1.0 / k, dtype=np.float32))
