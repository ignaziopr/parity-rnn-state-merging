import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class ParityRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParityRNN, self).__init__()
        self.hidden_size = hidden_size

        # Single RNN layer with ReLU
        self.rnn = nn.RNN(input_size, hidden_size,
                          nonlinearity='relu', batch_first=True)

        # Output/readout layer
        self.readout = nn.Linear(hidden_size, output_size)

        # Initialize weights as in the paper (Xavier with small gain)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=0.9)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=0.9)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.xavier_uniform_(self.readout.weight, gain=0.9)
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
        # Process packed sequence through RNN
        packed_output, final_hidden_state = self.rnn(packed_input)

        # Unpack to get padded output
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # Get the final valid output for each sequence in the batch
        batch_size = output.size(0)
        final_outputs = output[torch.arange(batch_size), lengths - 1]

        # Pass through readout layer
        logits = self.readout(final_outputs)

        return logits, final_outputs
