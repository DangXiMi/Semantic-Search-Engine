# src/embedder_onnx.py
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List
from src.config import settings
import json

class EmbedderONNX:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        onnx_path: str = "models/minilm-onnx/model.onnx",
        providers: list = None
    ):
        # ---- Model identifier for tokenizer ----
        self.model_name = model_name if model_name else settings.MODEL_NAME

        # ---- Load tokenizer (use HuggingFace model, not ONNX path) ----
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # ---- Load ONNX session ----
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # ---- Store input/output names for later ----
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name 

        # ---- Check output rank to decide if pooling is needed ----
        out_shape = self.session.get_outputs()[0].shape
        self.needs_pooling = (len(out_shape) == 3)  # True if (batch, seq, hidden)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convert texts to normalized embeddings using ONNX.
        Args:
            texts: list of strings
            batch_size: number of texts per inference batch
            show_progress: ignored for now (you can add tqdm later)
        Returns:
            np.ndarray of shape (len(texts), 384) – L2 normalized vectors
        """
        all_embeddings = []

        # ---- Batch loop ----
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # 1. Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np"
            )

            # 2. Run ONNX model (pass input_ids as int64)
            onnx_out = self.session.run(
                [self.output_name],
                {
                    "input_ids": encoded["input_ids"].astype(np.int64),
                    "attention_mask": encoded["attention_mask"].astype(np.int64)
                } )[0]
            # 3. Post-processing
            if self.needs_pooling:
                # Output shape: (batch, seq_len, hidden_dim)
                # Mean pooling with attention mask
                attention_mask = encoded["attention_mask"]
                # Expand mask to (batch, seq_len, 1) and cast to float
                mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
                # Apply mask and sum over sequence dimension
                pooled = np.sum(onnx_out * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1e-9)
            else:
                # Output already has shape (batch, hidden_dim) – no pooling needed
                pooled = onnx_out

            # 4. L2 normalization
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.maximum(norms, 1e-9)
           

            all_embeddings.append(normalized)

        return np.concatenate(all_embeddings, axis=0)

    # Optional: same convenience method as Embedder
    def encode_file(self, chunks_file: str, output_path: str, batch_size: int = 32):
        pass