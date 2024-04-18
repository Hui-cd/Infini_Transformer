import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadInfiniAttention(nn.Module):
    def __init__(self, n_head, dim_input, dim_k, dim_v, segment_length):
        super(MultiHeadInfiniAttention, self).__init__()
        self.n_head = n_head
        self.dim_input = dim_input
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.segment_length = segment_length
        self.beta = nn.Parameter(torch.zeros(1))

        self.w_q = nn.Linear(dim_input, n_head * dim_k)
        self.w_k = nn.Linear(dim_input, n_head * dim_k)
        self.w_v = nn.Linear(dim_input, n_head * dim_v)
        self.out = nn.Linear(n_head * dim_v, dim_input) 

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        n_seq, rem = divmod(seq_len, self.segment_length)
        if rem != 0:
            raise ValueError("Sequence length must be divisible by the segment length.")

        # Initialize memory and normalization factor
        mem = torch.zeros(batch_size, self.n_head, self.dim_k, self.dim_v, device=x.device)
        z = torch.zeros(batch_size, self.n_head, self.dim_k, 1, device=x.device)

        outputs = []
        for ix in range(n_seq):
            ix_lo = ix * self.segment_length
            ix_hi = ix_lo + self.segment_length
            segment = x[:, ix_lo:ix_hi, :]
            
            # Project segment to queries, keys, and values
            k = self.w_k(segment).view(batch_size, -1, self.n_head, self.dim_k).transpose(1, 2)
            v = self.w_v(segment).view(batch_size, -1, self.n_head, self.dim_v).transpose(1, 2)
            q = self.w_q(segment).view(batch_size, -1, self.n_head, self.dim_k).transpose(1, 2)

            # Update memory and normalization factor
            mem, z = self.memory_update(k, v, mem, z)

            # Compute attention

            att_dot = F.softmax(q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_k)), dim=-1) @ v
            att_mem = self.memory_retrieval(mem,z,q)

            # Weighted average of attentions
            att =self.long_term_context_injection(att_mem,att_dot,batch_size)
            # Append processed segment to outputs
            outputs.append(self.out(att))

        # Concatenate all segments to form the full sequence output
        return torch.cat(outputs, dim=1)

    def memory_retrieval(self, memory, z, q):
        sigma_q = F.elu(q) + 1.0
        a_mem = torch.matmul(sigma_q, memory) / (torch.matmul(sigma_q, z))
        return a_mem
    
    def memory_update(self, k, v, memory, z):
        sigma_k = F.elu(k) + 1.0
        sigma_k_transposed = sigma_k.transpose(-2, -1)  # Align dimensions for multiplication
        memory_update = torch.matmul(sigma_k_transposed, v)
        memory += memory_update
        summed_sigma_k = sigma_k.sum(dim=-2)
        z += summed_sigma_k.unsqueeze(-1) * self.segment_length  # Adjusting dimension if necessary
        return memory, z

    def long_term_context_injection(self, a_mem, a_dot,batch_size):
        att = F.sigmoid(self.beta) * a_mem + (1 - F.sigmoid(self.beta)) * a_dot
        att = att.view(batch_size, self.segment_length, self.n_head * self.dim_v)
        return att


if __name__ == '__main__':
    n_head = 8
    dim_input = 512
    dim_key = 64
    dim_value = 64
    segment_len = 32
    model = MultiHeadInfiniAttention(n_head=n_head, dim_input=dim_input, dim_k=dim_key, segment_length=segment_len, dim_v=dim_value)
    test = torch.randn(4, 128, dim_input)
    x = model(test)
