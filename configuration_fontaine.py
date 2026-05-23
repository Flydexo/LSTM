from transformers import PretrainedConfig


class FontaineConfig(PretrainedConfig):
    model_type = "fontaine"

    def __init__(
        self,
        vocab_size: int = 1500,
        n_hidden: int = 256,
        n_layers: int = 3,
        p_dropout: float = 0.4,
        seq_len: int = 72,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.seq_len = seq_len
