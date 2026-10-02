"""Task: adapt your RNN to use nn.LSTM.

The LSTM returns outputs, (hidden, cell). We can still classify outputs[:, -1, :].
Keep the input features, hidden width, output shape and training budget the same.
Explain what the additional cell state is for. Is improvement guaranteed here?
"""

from torch import nn


class StudentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, future):
        super().__init__()
        self.future = future
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future * 4)

    def forward(self, observations):
        outputs, _ = self.lstm(observations)
        last_output = outputs[:, -1, :]
        logits = self.fc(last_output)
        return logits.view(-1, self.future, 4)


ActionLSTM = StudentLSTM

if __name__ == "__main__":
    from supplied_pipeline import train_cli

    train_cli("lstm", ActionLSTM)
