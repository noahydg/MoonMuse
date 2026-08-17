import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoModel
from typing import Optional

# from .fast_transformer_decoder import FastTransformerDecoder
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from .transformer_helpers import (
  TokenEmbedding,
  PositionalEncoding,
  weights_init
)
# print ('[info] transformer_helpers imported')

def triangular_causal_mask(length, device):
  return torch.tril(torch.ones(length, length)).to(device)

class MusicGPT2(nn.Module):
  def __init__(self, n_token, n_layer, n_head, d_model, d_ff, d_embed,
    activation='relu', dropout=0.1, use_pe=True,
    use_segment_emb=False, n_segment_types=None,
    use_chord_mhot_emb=False
  ):
    super(MusicGPT2, self).__init__()
    self.n_token = n_token
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(d_model, n_token)

    # self.transformer_decoder = FastTransformerDecoder(
    #   n_layer, n_head, d_model, d_ff, dropout, activation, favor_feature_dims
    # )
    gpt_config = GPT2Config(
      n_layer=n_layer,
      n_head=n_head,
      n_embd=d_model,
      n_inner=d_ff,
      resid_pdrop=dropout,
      attn_pdrop=dropout,
      max_position_embeddings=4096,
    )
    self.transformer_decoder = nn.ModuleList([GPT2Block(gpt_config, layer_idx=i) for i in range(n_layer)])

    self.emb_dropout = nn.Dropout(self.dropout)
    self.use_pe = use_pe

    self.use_segment_emb = use_segment_emb
    if self.use_segment_emb:
      self.segemb = TokenEmbedding(n_segment_types, d_embed, d_model)
      self.n_segment_types = n_segment_types
    else:
      self.segemb = None

    self.use_chord_mhot_emb = use_chord_mhot_emb
    if use_chord_mhot_emb:
      self.chord_emb = nn.Linear(12, d_model)

    self.apply(weights_init)
    print ('[info] model init completed')

  def forward(self, x, seg_inp=None, chord_inp=None, keep_last_only=False, attn_kwargs=None):
    x_emb = self.token_emb(x)

    if seg_inp is not None and self.use_segment_emb:
      x_emb += self.segemb(seg_inp)

    if chord_inp is not None and self.use_chord_mhot_emb:
      x_emb += self.chord_emb(chord_inp)

    if self.use_pe:
      x_inp = self.emb_dropout(x_emb + self.pe(x.size(1)).permute(1, 0, 2))
    else:
      x_inp = self.emb_dropout(x_emb)

    dec_out = x_inp
    for i in range(self.n_layer):
      dec_out = self.transformer_decoder[i].forward(dec_out)[0]
    dec_logits = self.dec_out_proj(dec_out)

    if keep_last_only:
      dec_logits = dec_logits[:, -1, :]

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt, reduction='mean'):
    dec_tgt = dec_tgt.to(dtype=torch.long)
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction=reduction
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }

class ImprovedMusicGPT2(nn.Module):
    def __init__(
        self,
        n_token: int,
        n_layer: int,
        n_head: int,
        d_model: int,
        d_ff: int,
        d_embed: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        attention_window: int = 512,
        use_rope: bool = True,
        use_local_attn: bool = True,
    ):
        super().__init__()
        self.n_token = n_token
        self.d_model = d_model

        # 1. 输入嵌入层
        self.token_emb = nn.Embedding(n_token, d_embed)
        self.emb_proj = nn.Linear(d_embed, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_emb = nn.LayerNorm(d_model)  # Pre-LN

        # 2. 配置 RoPE + 局部注意力
        attention_types = self._get_attention_types(
            n_layer, use_local_attn, attention_window
        )
        config = GPTNeoConfig(
            num_layers=n_layer,
            num_heads=n_head,
            hidden_size=d_model,
            intermediate_size=d_ff,
            activation_function=activation,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            max_position_embeddings=max_seq_len,
            attention_types=attention_types,
            rotary=use_rope,  # 启用 RoPE
            window_size=attention_window,  # 局部注意力窗口大小
        )
        self.transformer = GPTNeoModel(config)

        # 3. 输出层
        self.head = nn.Linear(d_model, n_token)

        # 初始化权重
        self.apply(self._init_weights)

    def _get_attention_types(
        self, n_layer: int, use_local_attn: bool, window_size: int
    ) -> list[list]:
        """生成每层的注意力类型配置（符合HF GPTNeoConfig格式）"""
        if use_local_attn:
            # 前50%用global，后50%用local
            global_layers = n_layer // 2
            local_layers = n_layer - global_layers
            return [
                [["global"], global_layers],  # 全局注意力层配置
                [["local"], local_layers],    # 局部注意力层配置
            ]
        else:
            # 全部用global
            return [[["global"], n_layer]]

    def _init_weights(self, module):
        """权重初始化（与GPT-2类似）"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        seg_inp: Optional[torch.Tensor] = None,
        chord_inp: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        keep_last_only=False
    ) -> torch.Tensor:
        """
        Args:
            x: 输入token序列 [batch_size, seq_len]
            seg_inp: 可选的segment ID [batch_size, seq_len]
            chord_inp: 可选的chord多热编码 [batch_size, seq_len, 12]
            attention_mask: 自定义注意力掩码 [batch_size, seq_len]
        Returns:
            logits: 输出logits [batch_size, seq_len, n_token]
        """
        # 1. 输入嵌入
        x_emb = self.token_emb(x)  # [batch, seq_len, d_embed]
        
        # 可选：添加segment或chord嵌入
        if seg_inp is not None:
            seg_emb = self.token_emb(seg_inp)  # 复用token_emb
            x_emb = x_emb + seg_emb
        if chord_inp is not None:
            chord_emb = self.emb_proj(chord_inp.float())  # [batch, seq_len, d_model]
            x_emb = x_emb + chord_emb

        # 2. 投影到d_model + Pre-LN
        x_emb = self.emb_proj(x_emb)
        x_emb = self.ln_emb(x_emb)
        x_emb = self.emb_dropout(x_emb)

        # 3. Transformer（RoPE + 局部注意力）
        transformer_out = self.transformer(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
        ).last_hidden_state

        # 4. 输出层
        logits = self.head(transformer_out)
        
        if keep_last_only:
            logits = logits[:, -1, :]
            return logits

        return logits

    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        ignore_index: Optional[int] = None,
    ) -> dict:
        """计算交叉熵损失，并默认忽略本模型的 PAD token。"""
        if ignore_index is None:
            ignore_index = self.n_token - 1
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.contiguous().view(-1).long(), 
            ignore_index=ignore_index,
        )
        return {
          'recons_loss': loss,
          'total_loss': loss
        }


if __name__ == "__main__":
  # mask = triangular_causal_mask(100, "cpu")
  # print(mask.size(), mask[:10, :10])

  # bsize, seqlen = 2, 2048
  # model = MusicGPT2(100, 6, 4, 256, 2048, 256).to("cuda")

  # inp = torch.randint(0, 80, (bsize, seqlen)).to("cuda")
  # out = model.forward(inp)
  # print(out.size())

  bsize, seqlen = 2, 2048
  model = ImprovedMusicGPT2(100, 12, 8, 256, 2048, 256, attention_window=128, use_rope=True, use_local_attn=True).to("cuda")

  inp = torch.randint(0, 80, (bsize, seqlen)).to("cuda")
  out = model.forward(inp)
  print(out.size())
