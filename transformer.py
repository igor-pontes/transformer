import torch
from torch import nn
from functools import reduce

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  # MPS = Metal Performance Shaders
    if torch.backends.mps.is_available()
    else "cpu"
)


class AttentionLayer(nn.Module):
    def __init__(self, dim=512, d_k=512, d_v=512, mask=False):
        super().__init__()
        self.d_k = d_k
        self.W_q = (
            torch.rand(dim, d_k, requires_grad=True)
            if not mask
            else torch.triu(torch.rand(dim, d_k, requires_grad=True))
        )
        self.W_k = (
            torch.rand(dim, d_v, requires_grad=True)
            if not mask
            else torch.triu(torch.rand(dim, d_v, requires_grad=True))
        )
        self.W_v = (
            torch.rand(dim, d_v, requires_grad=True)
            if not mask
            else torch.triu(torch.rand(dim, d_v, requires_grad=True))
        )
        self.softmax = nn.Softmax()

    def forward(self, q, k, v):
        """
        Dimensions:
        (((n x dim) @ (dim x d_k)) @ ((N x dim) @ (dim x d_v)).t()) @ ((N x dim) @ (dim x d_v))
        ((n x d_k) @ (N x d_v).t()) @ (N x d_v)
        ((n x d_k) @ (d_v x N)) @ (N x d_v)
        (n x N) @ (N x d_v)
        n x d_v
        """
        return self.softmax(
            ((q @ self.W_q) @ (k @ self.W_k).t()) / torch.sqrt(torch.Tensor([self.d_k]))
        ) @ (v @ self.W_v)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, h=8, dim=512, mask=False):
        super().__init__()
        self.d = dim // h  # d_model/h
        self.W_o = torch.rand(dim, dim, requires_grad=True)
        self.heads = [AttentionLayer(512, self.d, self.d, mask) for _ in range(h)]

    def forward(self, Q, K, V):
        """
        Dimension
        (n x dim) @ (dim, dim)
        n x dim
        """
        return (
            reduce(
                lambda a, b: torch.cat(
                    (
                        a,
                        b(Q, K, V),
                    ),
                    1,
                ),
                self.heads[1:],
                self.heads[0](Q, K, V),
            )
            @ self.W_o
        )


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        in_dim=512,
        out_dim=512,
        inner_dim=2048,
    ):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, out_dim),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.mh_att_1 = MultiHeadAttentionLayer(dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.feed_forward = NeuralNetwork(in_dim=dim, out_dim=dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, Q, K, V):
        out1 = self.mh_att_1(Q, K, V) + Q
        norm1 = self.layer_norm1(out1)
        feed_1 = self.feed_forward(norm1) + out1
        return self.layer_norm2(feed_1)


class DecoderLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.mh_att_1 = MultiHeadAttentionLayer(dim=dim, mask=True)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.mh_att_2 = MultiHeadAttentionLayer(dim=dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.feed_forward = NeuralNetwork(in_dim=dim, out_dim=dim)
        self.layer_norm3 = nn.LayerNorm(dim)

    def forward(self, y, K_enc, V_enc):
        out1 = self.mh_att_1(y, y, y) + y
        norm1 = self.layer_norm1(out1)
        out2 = self.mh_att_2(norm1, K_enc, V_enc) + norm1
        feed_1 = self.feed_forward(out2) + out2
        return self.layer_norm2(feed_1)


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dim = dim
        self.encoder = EncoderLayer(dim)
        self.decoder = DecoderLayer(dim)

    def forward(self, x, y, vocab):
        embeds_x = self.get_embeddings(x, vocab) + self.pos_encoding(len(x))
        embeds_y = self.get_embeddings(y, vocab) + self.pos_encoding(len(y))

        enc_output_1 = self.encoder(embeds_x, embeds_x, embeds_x)
        dec_output_1 = self.decoder(embeds_y, enc_output_1, enc_output_1)

        enc_output_2 = self.encoder(enc_output_1, enc_output_1, enc_output_1)
        dec_output_2 = self.decoder(dec_output_1, enc_output_2, enc_output_2)

        enc_output_3 = self.encoder(enc_output_2, enc_output_2, enc_output_2)
        dec_output_3 = self.decoder(dec_output_2, enc_output_3, enc_output_3)

        enc_output_4 = self.encoder(enc_output_3, enc_output_3, enc_output_3)
        dec_output_4 = self.decoder(dec_output_3, enc_output_4, enc_output_4)

        enc_output_5 = self.encoder(enc_output_4, enc_output_4, enc_output_4)
        dec_output_5 = self.decoder(dec_output_4, enc_output_5, enc_output_5)

        enc_output_6 = self.encoder(enc_output_5, enc_output_5, enc_output_5)
        dec_output_6 = self.decoder(dec_output_5, enc_output_6, enc_output_6)

        print(f'{"-"*15} Encoder output {"-"*15}')
        print(enc_output_6)
        print(f'{"-"*15} Decoder output {"-"*15}')
        print(dec_output_6)

        return dec_output_6

    def get_embeddings(self, x, X):
        return self.embedding(torch.tensor([X[word] for word in x], dtype=torch.long))

    def sin_encoding(self, pos, i):
        return torch.sin(pos / (10000 ** ((2 * i) / self.dim)))

    def cos_encoding(self, pos, i):
        return torch.cos(pos / (10000 ** ((2 * i) / self.dim)))

    def pos_encoding(self, seq):
        out = torch.zeros(seq, self.dim)
        for p in torch.arange(seq):
            for i in torch.arange(self.dim):
                if i % 2 == 0:
                    out[p, i] += self.sin_encoding(p, i)
                    continue
                out[p, i] += self.cos_encoding(p, i)

        return out


def main():
    print(f"Using {device} device")
    vocab = {
        "<start>": 0,
        "I": 1,
        "am": 2,
        "a": 3,
        "great": 4,
        "cool": 5,
        "handsome": 6,
        "nice": 7,
        "cool!": 8,
        "guy": 9,
        "<end>": 10,
        "<SOS>": 11,
        "that": 12,
        "drinks": 13,
        "often": 14,
        "lot": 15,
        "and": 16,
        "smiles": 17,
    }

    input_text = "<start> I am a guy that drinks often <end>".split()
    output_text = "<start> cool and I am a guy that smiles".split()
    vocab_size = len(vocab)
    model = Transformer(vocab_size, 512)
    model(input_text, output_text, vocab)


if __name__ == "__main__":
    main()
