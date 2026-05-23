import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from configuration_fontaine import FontaineConfig


class FontaineLM(PreTrainedModel):
    config_class = FontaineConfig
    base_model_prefix = "lstm"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"fc.weight": None}  # v5: dict, key = weight to drop on save

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.fc

    def set_output_embeddings(self, value):
        self.fc = value

    def __init__(self, config: FontaineConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.n_hidden)
        self.lstm = nn.LSTM(
            input_size=config.n_hidden,
            hidden_size=config.n_hidden,
            num_layers=config.n_layers,
            dropout=config.p_dropout if config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.p_dropout)
        self.fc = nn.Linear(config.n_hidden, config.vocab_size, bias=False)
        self.fc.weight = self.embedding.weight  # weight tying
        self._hidden: tuple | None = None

    def reset_hidden(self):
        self._hidden = None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        hidden: tuple | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        bs = input_ids.size(0)

        if hidden is not None:
            h = hidden
        elif self._hidden is None or self._hidden[0].size(1) != bs:
            w = next(self.parameters()).data
            h = (
                w.new_zeros(self.config.n_layers, bs, self.config.n_hidden),
                w.new_zeros(self.config.n_layers, bs, self.config.n_hidden),
            )
        else:
            h = self._hidden

        emb = self.dropout(self.embedding(input_ids))
        out, new_h = self.lstm(emb, h)
        self._hidden = (new_h[0].detach(), new_h[1].detach())
        logits = self.fc(self.dropout(out))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels)

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_text(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        device: str | torch.device = "cpu",
    ) -> list[int]:
        self.eval()
        self.reset_hidden()
        device = torch.device(device)

        x = torch.tensor([prompt_ids], device=device)
        out = self(x)
        logits = out.logits[0, -1]

        generated = []
        for _ in range(max_new_tokens):
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
            generated.append(token)
            x = torch.tensor([[token]], device=device)
            logits = self(x).logits[0, -1]

        return generated
