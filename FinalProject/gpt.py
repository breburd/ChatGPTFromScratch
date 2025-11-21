import torch
import math
from rotary_position_embedding import RotaryPositionalEmbedding


class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((output_size, input_size)))
		self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

	def forward(self, x):
		return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((num_embeddings, embedding_dim)))

	def forward(self, x):
		return self.weight[x]


class CustomMHA(torch.nn.Module):

	def __init__(self, d_model, n_heads, max_seq_len=512):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
		self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
		self.dh = d_model//self.n_heads

		self.rope = RotaryPositionalEmbedding(self.dh, max_seq_len)

	def forward(self, x):
		added_batch = False
		if len(x.shape) == 2:
			added_batch = True
			x = x[None,:,:]

		# queries, keys, and values
		B, S, D = x.shape
		QKV = x @ self.qkv.T # B, S, 3D
		Q, K, V = torch.chunk(QKV, 3, -1)

		# split into multiple heads
		q_heads = torch.reshape(Q, (B, S, self.n_heads, self.dh))
		k_heads = torch.reshape(K, (B, S, self.n_heads, self.dh))
		v_heads = torch.reshape(V, (B, S, self.n_heads, self.dh))

		# Apply RoPE to q and k (not v) (https://huggingface.co/blog/designing-positional-encoding)
		q_heads = self.rope.forward(q_heads)  # shape stays: (B, S, n_heads, d_head)
		k_heads = self.rope.forward(k_heads)

		# reshape into (B*h, S, dh) so we isolate sequences for each head
		q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))
		k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))
		v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, self.dh))

		# make attention mask
		mask = torch.ones((S,S))
		mask = torch.tril(mask)
		mask = mask[None, :, :]
		mask = mask.to(x.device)

		# attention
		k_heads_t = torch.transpose(k_heads, 1, 2)
		qkt = torch.matmul(q_heads, k_heads_t) / math.sqrt(float(self.dh))
		qkt = qkt*mask
		qkt[qkt==0] = float('-inf')
		attn = torch.nn.functional.softmax(qkt, dim=-1)
		x = torch.matmul(attn, v_heads)

		# shmush back into the correct shape
		x = torch.reshape(x, (B, self.n_heads, S, self.dh))
		x = torch.transpose(x, 1, 2) # B, S, h, dh
		x = torch.reshape(x, (B, S, D))

		# apply projection
		x = x @ self.wo.T

		if added_batch:
			x = x[0]

		return x


class TransformerDecoderBlock(torch.nn.Module):

	def __init__(self, d_model, n_heads, max_seq_len=512):
		super().__init__()
		self.norm1 = torch.nn.LayerNorm((d_model,))
		self.mha = CustomMHA(d_model, n_heads, max_seq_len)
		self.norm2 = torch.nn.LayerNorm((d_model,))
		self.fc1 = CustomLinear(d_model, 4*d_model)
		self.act = torch.nn.ReLU()
		self.fc2 = CustomLinear(4*d_model, d_model)
		self.dropout = torch.nn.Dropout(0.1)

	def forward(self, x):
		x = x + self.mha(self.norm1(x))
		x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
		return x
		

class GPTModel(torch.nn.Module):

	def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
		super().__init__()

		self.word_embeddings = CustomEmbedding(vocab_size, d_model)

		self.layers = torch.nn.ModuleList()
		for i in range(layers):
			block = TransformerDecoderBlock(d_model, n_heads, max_seq_len)
			self.layers.append(block)

		self.fc_out = CustomLinear(d_model, vocab_size)

	def forward(self, x):
		B, S = x.shape
		positions = torch.arange(S).to(torch.long).to(x.device)
		positions = positions[None, :]
		positions = positions.repeat(B, 1)

		w_emb = self.word_embeddings(x)
		# no absolute position embeddings needed when using RoPE
		x = w_emb

		for layer in self.layers:
			x = layer(x)

		logits = self.fc_out(x)

		return logits


if __name__ == "__main__":

	model = GPTModel(128, 8, 4, 1000, 512)
	B = 32
	S = 48
	x = torch.randint(1000, (B, S))
	y = model(x)
