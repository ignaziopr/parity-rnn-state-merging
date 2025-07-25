import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class ParityRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParityRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size,
                          nonlinearity='relu', batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)

        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=0.99)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=0.99)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.99)
        nn.init.zeros_(self.readout.bias)

    def forward(self, packed_input):
        """
        Forward pass with packed sequences
        Args:
            packed_input: PackedSequence object
        Returns:
            logits: Final output logits (batch_size, output_size)
            final_hidden: Final hidden states (batch_size, hidden_size)
        """
        packed_output, final_hidden_state = self.rnn(packed_input)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        batch_size = output.size(0)
        final_outputs = output[torch.arange(batch_size), lengths - 1]
        logits = self.readout(final_outputs)

        return logits, final_outputs
