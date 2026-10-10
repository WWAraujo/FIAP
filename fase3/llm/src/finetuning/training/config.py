"""
Configurações para Fine-tuning de LLM Médico
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FineTuningConfig:
    """Configurações para fine-tuning"""

    # Modelo
    model_name: str = "gemma3:4b"  # Modelo base selecionado do benchmark (vencedor: p95=3538ms, 100% segurança, respostas em português)
    model_path: Optional[str] = None  # Caminho para modelo local se necessário

    # Diretórios
    output_dir: str = "./models/fine_tuned"
    cache_dir: str = "./models/cache"

    # Dados
    train_data_path: str = "./data/splits/train.jsonl"
    validation_data_path: str = "./data/splits/validation.jsonl"
    test_data_path: str = "./data/splits/test.jsonl"

    # Hiperparâmetros de treinamento
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Configurações do modelo
    max_seq_length: int = 512
    use_lora: bool = True  # Usar LoRA para fine-tuning eficiente
    lora_r: int = 8  # Rank do LoRA
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Será definido baseado no modelo

    # Otimização
    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Hardware
    use_fp16: bool = False
    use_bf16: bool = True  # BFloat16 para melhor precisão
    use_4bit: bool = True  # Quantização 4-bit para economia de memória

    # Logging
    logging_dir: str = "./logs"
    report_to: list = None  # ["tensorboard", "wandb"]

    # Validação
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    def __post_init__(self):
        """Define valores padrão após inicialização"""
        if self.lora_target_modules is None:
            # Módulos alvo comuns para modelos transformer
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        if self.report_to is None:
            self.report_to = ["tensorboard"]
