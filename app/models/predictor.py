import torch
import numpy as np
from app.core.config import settings

class Predictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (1,5,3) - input sequence of 5 points with x,y,t coordinates
        with torch.no_grad():
            inp = torch.from_numpy(x).float()
            out = self.model(inp)  # shape: (1, 10, 3)
            # Take the last predicted point (or first, depending on model design)
            return out[:, -1, :].numpy()  # shape: (1, 3)

def load_model():
    from app.models.network import TrajectoryPredictor
    
    # Create model with parameters matching the trained model
    model = TrajectoryPredictor(
        input_dim=3, 
        hidden_dim=settings.HIDDEN_SIZE, 
        output_dim=3, 
        num_layers=settings.NUM_LAYERS,
        dropout=0.5
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=settings.DEVICE))
    return Predictor(model)
