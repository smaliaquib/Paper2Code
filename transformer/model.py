import torch
import numpy as np
import torch.nn as nn

torch.manual_seed(321)

class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        # (seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros((seq_len, d_model))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term) # Even
        pe[:, 1::2] = torch.cos(position * div_term) # Odd
        pe = pe.unsqueeze(0)  # Add batch dimension

        self.register_buffer('pe', pe)  # Save as a non-trainable buffer

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + (self.pe[:, :x.size(1), :]) # (batch, seq_len, d_model)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads ==0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads ## d_k
        # d_k = d_v = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        ## Linear Projection for Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

        # Output Layer
        self.out = nn.Linear(d_model, d_model)

        # scaling
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, q, k, v, mask):
        
        batch = q.size(0)

        query = self.linear_q(q)
        key = self.linear_k(k)
        value = self.linear_v(v)

        # Split Q, K, V into multiple heads
        query = query.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (query @ key.transpose(-1, -2)) / self.scale
        # Scaled dot product attention
        if mask is not None:
            scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(scores)        
        attention_weights = self.dropout(attention_weights)
        attention_output = attention_weights @ value
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.head_dim)
 
        # Final linear transformation
        output = self.out(attention_output)
 
        return output


class LayerNormalization(nn.Module):

    def __init__(self, d_model, epsilon=1e-5):
        super().__init__()
        self.eps = epsilon
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # Compute mean and variance along the last dimension (feature dimension)
        mean = x.mean(dim=-1, keepdim=True) # Shape: (batch_size, seq_len, 1)
        variance = x.var(dim=-1, keepdim=True)

        # Normalized
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)

        # Scale and Shift
        output =  self.gamma * x_norm + self.beta
        return output


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.linear_w_1 = nn.Linear(d_model, d_hidden)
        self.linear_w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear_w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_w_2(x)
        return x


class ResidualConnection(nn.Module):

    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward Network
        self.ffn = PositionWiseFeedForward(d_model, d_hidden, dropout)

        # Residual Connections with Layer Normalization
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        # Apply residual connection for Multi-Head Attention
        x = self.residual1(x, lambda x: self.mha(x, x, x, mask)[0])  # Self-attention with residual

        # Apply residual connection for Feed-Forward Network
        x = self.residual2(x, self.ffn)
        
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, d_model, num_heads, d_hidden, vocab_size, max_len, dropout=0.1):
        super().__init__()

        # Input embedding + Positional Encoding
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stacked Encoder Blocks
        self.stacked_encoders = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_hidden, dropout) for _ in range(num_layers)
        ])

        # Final Normalization
        self.norm = LayerNormalization(d_model)

    def forward(self, x, src_mask=None):

        # Step 1: Embed the input
        x = self.embedding(x)

        # Step 2: Add Positional Encoding
        x = self.positional_encoding(x)

        # Step 3: Pass through all encoder blocks
        for enc_block in self.stacked_encoders:
            x = enc_block(x, src_mask)
        
        # Step 4: Finally Normalize it
        x = self.norm(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1):
        super().__init__()
        
        # Masked Self-Attention
        self.mmha = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)

        # Encoder-Decoder Attention
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

        # Feed-Forward Network
        self.ffn = PositionWiseFeedForward(d_model, d_hidden, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked MultiHead-Attention with Residual Connection
        tgt = self.residual1(tgt, lambda x: self.mmha(x, x, x, tgt_mask)[0])
        
        # Encoder-Decoder Attention with Residual Connection
        tgt = self.residual2(tgt, lambda x: self.encoder_decoder_attention(x, memory, memory, memory_mask)[0])
        
        # Feed-Forward Network with Residual Connection
        tgt = self.residual3(tgt, self.ffn)
        
        return tgt
    

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_hidden, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Input Embedding + Positional Encoding
        self.embedding = InputEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stacked Decoder Blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_hidden, dropout) for _ in range(num_layers)
        ])
        
        # Final Linear Layer for Vocabulary Projection
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Step 1: Embed the target input
        x = self.embedding(tgt)  # Shape: [batch_size, tgt_seq_len, d_model]
        
        # Step 2: Add Positional Encoding
        x = self.positional_encoding(x)  # Shape: [batch_size, tgt_seq_len, d_model]
        
        # Step 3: Pass through each Decoder Block
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, memory, tgt_mask, memory_mask)
        
        # Step 4: Project to vocabulary size
        logits = self.linear(x)  # Shape: [batch_size, tgt_seq_len, vocab_size]
        
        return logits

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 src_max_seq_len, 
                 tgt_max_seq_len, 
                 d_model, 
                 num_heads, 
                 d_hidden, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_hidden, src_vocab_size, src_max_seq_len, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_hidden, tgt_vocab_size, tgt_max_seq_len, dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Encode source sequence
        memory = self.encoder(src, src_mask)
        
        # Decode target sequence
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        
        return output



def create_padding_mask(seq, pad_token):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]

def create_causal_mask(seq_len):
    return torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()  # Shape: [seq_len, seq_len]


import torch.optim as optim

# Hyperparameters
src_vocab_size = 5000
tgt_vocab_size = 5000
src_max_seq_len = 100
tgt_max_seq_len = 100
d_model = 512
num_heads = 8
d_hidden = 2048
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.1
pad_token = 0

# Initialize the Transformer
model = Transformer(src_vocab_size, tgt_vocab_size, src_max_seq_len, tgt_max_seq_len, d_model, num_heads, d_hidden, num_encoder_layers, num_decoder_layers, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)  # Ignore pad tokens in loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Dummy dataset (replace with real data)
src_batch = torch.randint(1, src_vocab_size, (32, src_max_seq_len))  # [batch_size, src_seq_len]
tgt_batch = torch.randint(1, tgt_vocab_size, (32, tgt_max_seq_len))  # [batch_size, tgt_seq_len]
tgt_input = tgt_batch[:, :-1]
tgt_output = tgt_batch[:, 1:]

# Generate masks
src_mask = create_padding_mask(src_batch, pad_token)
tgt_mask = create_padding_mask(tgt_input, pad_token) & create_causal_mask(tgt_input.size(1))
memory_mask = None

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(src_batch, tgt_input, src_mask, tgt_mask, memory_mask)  # Shape: [batch_size, tgt_seq_len, tgt_vocab_size]
    output = output.reshape(-1, tgt_vocab_size)  # Flatten for loss calculation
    tgt_output = tgt_output.reshape(-1)  # Flatten target

    # Compute loss
    loss = criterion(output, tgt_output)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")



