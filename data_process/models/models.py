import torch
import torch.nn as nn

class HandInterfaceTransformer(torch.nn.Module):
    '''
    A model to generate a hand pose given an object point cloud by outputting a hand interface.
    The model architecture is based on a transformer encoder-decoder structure.

    input: 
    - object_pc: torch.Tensor of shape (batch_size, num_points, 3)
    - k_query: int, number of queries to generate for the hand interface

    output:
    - hand_interface: torch.Tensor of shape (batch_size, k_query, dim_interface)
      where dim_interface is the dimension of the hand interface representation.

    pipeline:
    1. Encode the object point cloud using a predefined encoder (which is initialized in __init__).
    2. sample K query tokens from the token pool, each token is a vector of shape (dim_model,).
        The token pool is initialized in the constructor and contains a fixed number of tokens.
        The sampling is done by randomly selecting K indices.
        The sampled query tokens are then passed to the transformer decoder.
    3. Decode the sampled query tokens using a transformer decoder to generate the hand interface, 
        the encoded object point cloud is used as the context for the decoder.
    4. Decode the output of the transformer decoder to the final hand interface representation.
       The final output is a tensor of shape (batch_size, k_query, dim_interface).
    '''

    def __init__(self, 
                 dim_model=512, 
                 num_heads=8, 
                 num_layers=6, 
                 dim_feedforward=2048, 
                 dim_interface=12,
                 dropout=0.1, 
                 k_query=10,
                 ):
        super(HandInterfaceTransformer, self).__init__()
        
        self.dim_model = dim_model
        self.k_query = k_query
        
        # Token pool for queries
        self.token_pool = torch.nn.Embedding(k_query, dim_model)
        
        # Transformer decoder
        self.transformer_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        
        # Final linear layer to output the hand interface
        self.output_layer = torch.nn.Linear(dim_model, dim_interface)

    def forward(self, object_pc: torch.Tensor):
        """
        Forward pass of the model.
        
        Args:
            object_pc (torch.Tensor): Object point cloud of shape (batch_size, num_points, 3).
        
        Returns:
            torch.Tensor: Hand interface of shape (batch_size, k_query, dim_interface).
        """
        batch_size = object_pc.shape[0]
        
        sampled_queries = self.token_pool.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (k_query, batch_size, dim_model)
        encoded_object = encoded_object.transpose(0, 1)  # (num_points, batch_size, dim_model)  
        # Pass through the transformer decoder
        transformer_output = self.transformer_decoder(
            sampled_queries, 
            encoded_object, 
            memory_key_padding_mask=None,  # Assuming no padding in the object point cloud
            tgt_key_padding_mask=None  # Assuming no padding in the queries
        )
        # transformer_output is of shape (k_query, batch_size, dim_model)

        hand_interface = self.output_layer(transformer_output)
        # hand_interface is of shape (k_query, batch_size, dim_interface)
        hand_interface = hand_interface.transpose(0, 1)  # (batch_size, k_query, dim_interface)

        return hand_interface


