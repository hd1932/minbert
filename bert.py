import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, uses the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    # - Normalizes the scores with softmax.
    # - Multiplies the attention scores with the value to get back weighted values.
    # - Before returning, concatenates multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].
    
    batch_size, num_attention_heads, seq_length, attention_head_size = key.size()
    attention_scores = (torch.matmul(query, key.transpose(-1, -2)) / (attention_head_size ** 0.5)) + attention_mask
    # Normalize for probs.
    attention_probs = F.softmax(attention_scores, dim=-1)
    # Score * value = context layer
    # Transpose
    context_layer = torch.matmul(attention_probs, value).transpose(1, 2).contiguous()
    # Reshape the context layer back
    context_layer = context_layer.view(batch_size, seq_length, -1)
    return context_layer

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    return ln_layer(input + dropout(dense_layer(output)))
  

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    attention_output = self.add_norm(
          hidden_states, 
          self.self_attention(hidden_states, attention_mask), 
          self.attention_dense, 
          self.attention_dropout, 
          self.attention_layer_norm
    )
    layer_output = self.add_norm(
        attention_output, 
        self.interm_af(self.interm_dense(attention_output)), 
        self.out_dense, 
        self.out_dropout, 
        self.out_layer_norm
    )
    return layer_output

class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]
    inputs_embeds = self.word_embedding(input_ids)
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    embeddings = inputs_embeds + pos_embeds + tk_type_embeds
    embeddings = self.embed_layer_norm(embeddings)
    embeddings = self.embed_dropout(embeddings)
    return embeddings


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    for i, layer_module in enumerate(self.bert_layers):
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    embedding_output = self.embed(input_ids=input_ids)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
