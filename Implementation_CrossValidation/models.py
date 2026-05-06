import torch
import torch.nn as nn


def combine_sensor_streams(acc, gyro):
    return torch.cat([acc, gyro], dim=2)


def reduced_time_dim(window_size):
    length = ((window_size + 2 * 2 - 6) // 2) + 1
    length = ((length - 2) // 2) + 1
    length = ((length + 2 * 1 - 3) // 2) + 1
    length = ((length - 2) // 2) + 1
    return length


class TilleyCNNExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=24,
                kernel_size=6,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.conv_blocks(x)


class TilleyCNNOnly(nn.Module):
    def __init__(
        self,
        num_classes=12,
        input_channels=6,
        window_size=128,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.cnn = TilleyCNNExtractor(input_channels=input_channels)
        flattened_dim = 48 * reduced_time_dim(window_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_dim, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes),
        )

    def forward(self, acc, gyro):
        x = combine_sensor_streams(acc, gyro)
        x = self.cnn(x)
        return self.classifier(x)


class TilleyLSTMOnly(nn.Module):
    def __init__(
        self,
        num_classes=12,
        input_channels=6,
        hidden_size=24,
        num_layers=1,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes),
        )

    def forward(self, acc, gyro):
        x = combine_sensor_streams(acc, gyro)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


class TilleyCNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=12,
        input_channels=6,
        hidden_size=24,
        num_layers=1,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.cnn = TilleyCNNExtractor(input_channels=input_channels)
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, acc, gyro):
        x = combine_sensor_streams(acc, gyro)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


class CNNHead(nn.Module):
    """
    One CNN head for one sensor stream.

    Input shape:
        (batch, window_size, input_channels)

    Output shape:
        (batch, window_size, 48)
    """

    def __init__(self, input_channels=3, window_size=128, dropout_rate=0.5):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=24,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            nn.LayerNorm(window_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(
                in_channels=24,
                out_channels=48,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            nn.LayerNorm(window_size),
            nn.Dropout(dropout_rate),

            nn.Conv1d(
                in_channels=48,
                out_channels=48,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            nn.LayerNorm(window_size),
        )

    def forward(self, x):
        # x starts as (batch, time, axes)
        # Conv1d needs (batch, channels, time)
        x = x.permute(0, 2, 1)

        x = self.conv_block(x)

        # return to (batch, time, features)
        x = x.permute(0, 2, 1)

        return x


class MultiheadCNNLSTM(nn.Module):
    """
    PyTorch implementation of the previous TensorFlow Multihead CNN-LSTM.

    Inputs:
        acc:  (batch, window_size, input_channels)
        gyro: (batch, window_size, input_channels)

    Output:
        logits: (batch, 12)

    Important:
        No softmax is applied here.
        PyTorch CrossEntropyLoss expects raw logits.
    """

    def __init__(
        self,
        num_classes=12,
        input_channels=3,
        window_size=128,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.acc_head = CNNHead(
            input_channels=input_channels,
            window_size=window_size,
            dropout_rate=dropout_rate,
        )

        self.gyro_head = CNNHead(
            input_channels=input_channels,
            window_size=window_size,
            dropout_rate=dropout_rate,
        )

        # Each CNN head outputs 48 features.
        # After concatenation: 48 + 48 = 96 features per timestep.
        self.lstm1 = nn.LSTM(
            input_size=96,
            hidden_size=4 * num_classes,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(4 * num_classes)

        self.lstm2 = nn.LSTM(
            input_size=4 * num_classes,
            hidden_size=4 * num_classes,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(4 * num_classes)

        self.lstm3 = nn.LSTM(
            input_size=4 * num_classes,
            hidden_size=2 * num_classes,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(2 * num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(2 * num_classes, 2 * num_classes),
            nn.Linear(2 * num_classes, num_classes),
        )

    def forward(self, acc, gyro):
        acc_features = self.acc_head(acc)
        gyro_features = self.gyro_head(gyro)

        x = torch.cat([acc_features, gyro_features], dim=2)

        x, _ = self.lstm1(x)
        x = self.norm1(x)

        x, _ = self.lstm2(x)
        x = self.norm2(x)

        x, _ = self.lstm3(x)
        x = self.norm3(x)

        # Use only the final timestep output
        x = x[:, -1, :]

        logits = self.classifier(x)

        return logits


def build_model(config, type, hp = None):
    hp = hp or {}
    combined_input_channels = config.NUM_SENSORS * config.NUM_AXES

    if type == config.MulitHeadCNNLSTM_type:
        return MultiheadCNNLSTM(
            num_classes=config.NUM_CLASSES,
            input_channels=config.NUM_AXES,
            window_size=config.WINDOW_SIZE,
            dropout_rate=hp.get("dropout_rate", 0.5),
        )

    if type == config.CNN_Type:
        return TilleyCNNOnly(
            num_classes=config.NUM_CLASSES,
            input_channels=combined_input_channels,
            window_size=config.WINDOW_SIZE,
            dropout_rate=hp.get("dropout_rate", 0.5),
        )

    if type == config.LSTM_Type:
        return TilleyLSTMOnly(
            num_classes=config.NUM_CLASSES,
            input_channels=combined_input_channels,
            hidden_size=hp.get("hidden_size", 24),
            num_layers=config.LSTM_NUM_LAYERS,
            dropout_rate=hp.get("dropout_rate", 0.5),
        )

    if type == config.CNNLSTM_Type:
        return TilleyCNNLSTM(
            num_classes=config.NUM_CLASSES,
            input_channels=combined_input_channels,
            hidden_size=hp.get("hidden_size", 24),
            num_layers=config.LSTM_NUM_LAYERS,
            dropout_rate=hp.get("dropout_rate", 0.5),
        )

    raise ValueError(f"Unsupported model type: {type}")
