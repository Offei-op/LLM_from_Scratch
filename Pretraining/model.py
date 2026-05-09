import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x:torch.tensor):
        mean = x.mean(dim = -1,keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x-mean)/(var**0.5 + self.eps)
        x = x*self.scale + self.shift
        return x
    

class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))
                                         * (x+ 0.044715*torch.pow(x,3))))
    

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.ffn = nn.Sequential(nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                                 GELU(),
                                 nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]))
        

    def forward(self,x):
        return self.ffn(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,num_heads,qkv_bias = False):
        super().__init__()
        assert(d_out%num_heads == 0)
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.out_proj = nn.Linear(d_out,d_out,bias=qkv_bias)
        

    def forward(self,x):
        b,token_length,d_in = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        q_s = Q.view(b,token_length,self.num_heads,self.head_dim) 
        k_s = K.view(b,token_length,self.num_heads,self.head_dim)
        v_s = V.view(b,token_length,self.num_heads,self.head_dim)


        q_s = q_s.transpose(1,2)
        k_s = k_s.transpose(1,2)
        v_s = v_s.transpose(1,2)

        attention_scores = q_s @ k_s.transpose(2,3)
        mask_bool = self.mask[:token_length,:token_length].bool()
        attention_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attention_scores/k_s.shape[-1]**0.5,dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = (attn_weights @ v_s).transpose(1,2)
        context_vecs = context_vecs.contiguous().view(b,token_length,self.d_out)
        context_vecs = self.out_proj(context_vecs)
        return context_vecs
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.mha = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['mha_drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )
        self.FFN = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg['shortcut_drop_rate'])


    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.FFN(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['embedding_drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias=False)
        

    def forward(self,inp_x):
        batch_size, seq_len = inp_x.shape
        tok = self.tok_emb(inp_x)
        pos = self.pos_emb(torch.arange(seq_len, device=inp_x.device))

        x = tok + pos
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits