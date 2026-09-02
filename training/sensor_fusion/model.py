import torch
import torch.nn as nn

class IMUDriftPredictor(nn.Module):
    """
    CNN + BiLSTM neural network for predicting IMU drift errors.
    
    Predicts position and velocity drift [δp_N, δp_E, δp_D, δv_N, δv_E, δv_D]
    from a window of IMU data.
    """
    def __init__(
        self,
        input_features: int = 9,
        seq_len: int = 50,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # CNN Feature Extractor
        # Input shape needs to be (batch, input_features, seq_len)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Fully Connected Layers
        # Bi-LSTM output is hidden_size * 2
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_features).
            
        Returns:
            torch.Tensor: Predicted drift (batch, 6).
        """
        # Permute for Conv1D: (batch, input_features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN
        x = self.conv_layers(x)
        
        # Permute back for LSTM: (batch, seq_len, 64)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # Using the hidden state from the last timestep
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # We want the last layer's hidden states for both directions
        h_last_fwd = h_n[-2]
        h_last_bwd = h_n[-1]
        
        # Concatenate forward and backward
        lstm_out = torch.cat((h_last_fwd, h_last_bwd), dim=1)
        
        # FC
        out = self.fc_layers(lstm_out)
        
        return out

class DriftPredictionLoss(nn.Module):
    """
    Weighted MSE loss for drift prediction.
    L = λ_p * MSE(δp_pred, δp_true) + λ_v * MSE(δv_pred, δv_true)
    """
    def __init__(self, lambda_p: float = 1.0, lambda_v: float = 0.5):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_v = lambda_v
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred (torch.Tensor): Predicted tensor (batch, 6).
            target (torch.Tensor): Target tensor (batch, 6).
            
        Returns:
            torch.Tensor: Loss scalar.
        """
        # First 3 are position, next 3 are velocity
        pos_pred = pred[:, :3]
        pos_target = target[:, :3]
        
        vel_pred = pred[:, 3:]
        vel_target = target[:, 3:]
        
        pos_loss = self.mse(pos_pred, pos_target)
        vel_loss = self.mse(vel_pred, vel_target)
        
        return self.lambda_p * pos_loss + self.lambda_v * vel_loss

def create_model(input_features: int = 9, seq_len: int = 50, **kwargs) -> IMUDriftPredictor:
    """
    Factory function to create IMUDriftPredictor model.
    
    Args:
        input_features (int): Number of input features.
        seq_len (int): Sequence length.
        **kwargs: Additional arguments for the model.
        
    Returns:
        IMUDriftPredictor: Instantiated model.
    """
    return IMUDriftPredictor(input_features=input_features, seq_len=seq_len, **kwargs)

def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model.
        
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Instantiate model
    model = create_model()
    
    # Print parameter count
    print(f"Total trainable parameters: {count_parameters(model)}")
    
    # Forward pass with dummy data
    batch_size = 4
    seq_len = 50
    features = 9
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Dummy loss
    dummy_target = torch.randn(batch_size, 6)
    criterion = DriftPredictionLoss()
    loss = criterion(output, dummy_target)
    print(f"Dummy loss: {loss.item()}")
