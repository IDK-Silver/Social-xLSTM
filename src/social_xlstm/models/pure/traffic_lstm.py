import torch
import torch.nn as nn

class TrafficLSTM(nn.Module):
    """LSTM model for traffic prediction for a single VD."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        prediction_length,
        dropout=0.2
    ):
        """
        Initialize LSTM model for traffic prediction.

        Args:
            input_size (int): Number of input features for the VD (e.g., flow, speed).
            hidden_size (int): LSTM hidden size.
            num_layers (int): Number of LSTM layers.
            prediction_length (int): Number of future timesteps to predict.
            dropout (float): Dropout rate.
        """
        super(TrafficLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # Input and output tensors have batch dimension first
            dropout=dropout if self.num_layers > 1 else 0
        )

        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),  # Linear layer
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(self.hidden_size // 2, self.input_size * self.prediction_length)  # Linear layer
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, seq_len, num_features]
                              batch_size: batch size
                              seq_len: sequence length (number of historical timesteps)
                              num_features: number of features per timestep (should equal self.input_size)

        Returns:
            torch.Tensor: predictions with shape [batch_size, prediction_length, num_features]
                          prediction_length: number of future timesteps to predict
                          num_features: number of features to predict (should equal self.input_size)
        """
        batch_size, seq_len, num_features = x.shape

        # Initialize hidden states and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        # x shape is already [batch_size, seq_len, num_features], can be directly input to LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take the last output from the sequence
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        last_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_size]

        # Project to prediction
        # predictions shape: [batch_size, input_size * prediction_length]
        predictions = self.output_projection(last_output)

        # Reshape to final output format
        # [batch_size, prediction_length, num_features]
        predictions = predictions.view(
            batch_size, self.prediction_length, self.input_size
        )

        return predictions