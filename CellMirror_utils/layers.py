import collections
import torch
import torch.nn as nn
from torch.autograd.variable import Variable

class build_multi_layers(nn.Module):

    def __init__(
        self, 
        n_in: int, n_out: int, n_hidden: list, 
        dropout_rate: float = 0.1, use_batch_norm: bool = True, 
        use_relu: bool = True, 
        use_sigmoid: bool = False, 
        bias: bool = True
    ):
        super().__init__()
        layers_dim = [n_in] + n_hidden

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out, bias=bias),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.ReLU() if use_relu else None,
                            nn.Sigmoid() if use_sigmoid else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate( zip( layers_dim[:-1], layers_dim[1:] ) )
                ]
            )
        )

    def forward(self, x: torch.Tensor):

        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    x = layer(x)

        return x

class Encoder(nn.Module):
    def __init__(
        self, 
        n_input: int, 
        n_output: int, 
        n_hidden: list, 
        dropout_rate = 0
    ):
        super(Encoder, self).__init__()

        self.encoder = build_multi_layers(n_in = n_input, n_out = n_output, n_hidden = n_hidden, dropout_rate = dropout_rate)

        self.mean_encoder = nn.Linear(n_hidden[-1], n_output)
        self.log_var_encoder = nn.Linear(n_hidden[-1], n_output)

    def reparameterize_gaussian(self, mu, var):

        std = var.sqrt()
        eps = Variable(std.data.new(std.size()).normal_())

        return eps.mul(std).add_(mu)

    def forward(self, x: torch.Tensor):

        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.log_var_encoder(q)) 
        latent = self.reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent

class Decoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: list,
        n_output: int,
        dropout_rate=0
    ):
        super(Decoder, self).__init__()

        self.decoder = build_multi_layers(n_in = n_input, n_out = n_output, n_hidden = n_hidden, dropout_rate = dropout_rate)
        self.decoder_x = nn.Linear( n_hidden[-1], n_output )

    def forward(self, z: torch.Tensor):
        outputs = self.decoder(z)
        outputs = self.decoder_x(outputs)

        return outputs

class LinearDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: list,
        n_output: int,
        bias: bool = False,
        dropout_rate = 0
    ):
        super(LinearDecoder, self).__init__()

        if len(n_hidden) > 0:
            self.factor_regressor = build_multi_layers(
                n_in = n_input,
                n_out = n_output,
                n_hidden = n_hidden,
                use_relu = False,
                bias = bias,
                dropout_rate = dropout_rate
            )

            self.decoder_x = nn.Linear( n_hidden[-1], n_output, bias=bias)
        else:
            self.factor_regressor = build_multi_layers(
                n_in = n_input,
                n_out = n_output,
                n_hidden = [n_output],
                use_relu = False,
                bias = bias,
                dropout_rate = dropout_rate
            )

        self.n_hidden = n_hidden

    def forward(self, z: torch.Tensor):
        if len(self.n_hidden) > 0:
            outputs = self.factor_regressor(z)
            outputs = self.decoder_x(outputs)

        else:
            outputs = self.factor_regressor(z)

        return outputs

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Discriminator(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int = 1,
        use_sigmoid: bool = True,
        dropout_rate = 0
    ):
        super(Discriminator, self).__init__()

        self.discriminator = build_multi_layers(
            n_in = n_input,
            n_out = n_output,
            n_hidden = [n_output],
            use_sigmoid = use_sigmoid,
            use_relu = False,
            dropout_rate = dropout_rate
        )

    def forward(self, z:torch.Tensor):

        score = self.discriminator(z)

        return score