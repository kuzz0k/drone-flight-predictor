import torch
import numpy as np
from app.core.config import settings

class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (1,5,3)
        with torch.no_grad():
            inp = torch.from_numpy(x).float()
            out = self.model(inp)
        return out.numpy()

def load_model():
    from app.models.network import LSTMNetwork
    net = LSTMNetwork(input_size=3, hidden_size=settings.HIDDEN_SIZE, num_layers=settings.NUM_LAYERS)
    net.load_state_dict(torch.load(settings.MODEL_PATH, map_location=settings.DEVICE))
    return Predictor(net)
