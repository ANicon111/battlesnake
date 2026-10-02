"""Task: define an RNN and a linear layer to predict the next x actions.

Input: (batch, HISTORY, number of features).
1. Use nn.RNN(..., batch_first=True).
2. Select the last output: outputs[:, -1, :].
3. Apply nn.Linear(hidden_size, future * 4).
4. Reshape to (batch, future, 4). Return LOGITS, not softmax probabilities.
The supplied pipeline handles training, validation and plotting.
"""

from torch import nn


class StudentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, future):
        super().__init__()
        self.future = future
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, future * 4)

    def forward(self, observations):
        outputs, _ = self.rnn(observations)
        last_output = outputs[:, -1, :]
        logits = self.fc(last_output)
        return logits.view(-1, self.future, 4)


ActionRNN = StudentRNN

if __name__ == "__main__":
    from supplied_pipeline import train_cli

    train_cli("rnn", ActionRNN)
