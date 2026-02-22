"""LLaVA-Med v1.5 Mistral-7B inference backend."""

import itertools
import json
import logging
import os
import time

import torch
from PIL import Image

from radiology_vqa.vlm.interface import VLMPrediction

logger = logging.getLogger(__name__)

# ── Weight key remapping ──────────────────────────────────────────────────────
# transformers ≥ 4.45 renamed LlavaForConditionalGeneration's expected weight
# paths. The LLaVA-Med checkpoint was released with the OLD flat naming
# convention. We remap old → new so weights land in the right model positions.
#
# Detected by checking for "model.embed_tokens.weight" in the checkpoint index:
#   Old format (checkpoint) → New format (transformers ≥ 4.45)
#   model.layers.*                    → model.language_model.layers.*
#   model.embed_tokens.*              → model.language_model.embed_tokens.*
#   model.norm.*                      → model.language_model.norm.*
#   model.vision_tower.vision_tower.* → model.vision_tower.*
#   model.mm_projector.0.*            → model.multi_modal_projector.linear_1.*
#   model.mm_projector.2.*            → model.multi_modal_projector.linear_2.*
# lm_head.* is unchanged in both formats.

_OLD_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    # Vision tower: strip the extra nesting level introduced by the old code
    ("model.vision_tower.vision_tower.", "model.vision_tower."),
    # Projector: renamed and split into linear_1 / linear_2
    ("model.mm_projector.0.", "model.multi_modal_projector.linear_1."),
    ("model.mm_projector.2.", "model.multi_modal_projector.linear_2."),
    # Language model: add the language_model. namespace
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
)


def _remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate old-format LLaVA-Med checkpoint keys to new transformers paths."""
    remapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _OLD_KEY_PREFIXES:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = val
    return remapped


def _remap_key(key: str) -> str:
    """Remap a single checkpoint key from old to new format (no-alloc fast path)."""
    for old_prefix, new_prefix in _OLD_KEY_PREFIXES:
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def _set_param_direct(
    model: torch.nn.Module,
    key: str,
    tensor: torch.Tensor,
    device: torch.device,
) -> None:
    """Set a model parameter by dotted name without accelerate.

    Works for non-meta parameters only.  Used as a fallback when accelerate
    is not installed (CPU-only environments where [quant] was not installed).
    """
    parts = key.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    attr = parts[-1]
    current = getattr(module, attr)
    if isinstance(current, torch.nn.Parameter):
        setattr(module, attr, torch.nn.Parameter(tensor.to(device)))
    else:
        current.data.copy_(tensor.to(device))


class LLaVAMedBackend:
    """LLaVA-Med v1.5 (Mistral-7B) inference backend.

    Uses LlavaForConditionalGeneration from transformers.
    Supports 4-bit / 8-bit quantization via bitsandbytes (CUDA only).
    Falls back to fp32 on CPU automatically.

    Compatibility note
    ------------------
    The ``microsoft/llava-med-v1.5-mistral-7b`` checkpoint was released with an
    older weight naming convention (flat ``model.layers.*``) that diverged from
    ``transformers >= 4.45`` which expects nested ``model.language_model.layers.*``.
    This backend detects the mismatch automatically and applies key remapping,
    falling back from bitsandbytes quantization to fp16 when required (bitsandbytes
    quantization must happen during ``from_pretrained`` and cannot be applied to a
    manually-loaded state dict without re-quantising every tensor).
    To keep 4-bit quantization, pin ``transformers==4.44.2``.
    """

    def __init__(
        self,
        model_id: str = "microsoft/llava-med-v1.5-mistral-7b",
        quantize: str = "4bit",
        device: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens

        cuda_available = torch.cuda.is_available()
        if not cuda_available and quantize != "none":
            logger.warning(
                "CUDA not available but quantize=%r requested. "
                "Falling back to fp32 on CPU — inference will be slow and may OOM.",
                quantize,
            )
            self._quantize = "none"
        else:
            self._quantize = quantize

        self._device = "cpu" if not cuda_available else device

        logger.info(
            "Loading LLaVA-Med: model=%s quantize=%s device=%s",
            model_id,
            self._quantize,
            self._device,
        )
        self._processor, self._model = self._load_model(model_id)
        # Cache the resolved device once — avoids walking the parameter
        # iterator on every predict() call.
        self._inferred_device: torch.device = next(self._model.parameters()).device
        logger.info("LLaVA-Med loaded on device=%s.", self._inferred_device)

    # ── model loading ─────────────────────────────────────────────────────────

    def _detect_old_key_format(self, model_id: str) -> bool:
        """Return True if the checkpoint uses the old flat LLaVA weight naming.

        Reads only the lightweight ``model.safetensors.index.json`` (73 KB,
        always downloaded first and already in the HF cache). Does not touch
        the multi-GB weight shards.
        """
        try:
            from huggingface_hub import hf_hub_download

            index_path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                local_files_only=True,  # use cache; don't re-download
            )
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map: dict[str, str] = index.get("weight_map", {})
            # Old format: "model.embed_tokens.weight"   (flat, no language_model. ns)
            # New format: "model.language_model.embed_tokens.weight"
            old = "model.embed_tokens.weight" in weight_map
            if old:
                logger.warning(
                    "Checkpoint '%s' uses OLD flat LLaVA weight key format "
                    "(model.embed_tokens.weight, model.layers.*, …). "
                    "transformers >= 4.45 expects nested "
                    "model.language_model.* paths. "
                    "Applying weight key remapping automatically.",
                    model_id,
                )
            return old
        except Exception as exc:
            logger.debug("Could not detect checkpoint key format: %s", exc)
            return False

    def _load_checkpoint_state_dict(self, model_id: str) -> dict[str, torch.Tensor]:
        """Read all safetensors shards from the HF cache into a single dict."""
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for weight key remapping. "
                "Run: pip install safetensors"
            ) from exc

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_id, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot locate cached checkpoint for '{model_id}'. "
                f"Ensure the model has been downloaded. Error: {exc}"
            ) from exc

        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))
        logger.info("Remapping weights from %d shard(s)…", len(shard_files))

        state_dict: dict[str, torch.Tensor] = {}
        for shard_file in shard_files:
            shard_path = os.path.join(local_dir, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        return state_dict

    def _apply_remapped_weights(self, model: torch.nn.Module, model_id: str) -> None:
        """Load checkpoint weights into *model* using shard-by-shard streaming.

        Memory-safe approach for T4 (16 GB VRAM):
        - Processes ONE safetensors shard (~3.5 GB) at a time; frees it before
          loading the next — peak CPU RAM ≈ 3.5 GB.
        - Uses ``set_module_tensor_to_device`` to replace each GPU parameter
          IN-PLACE: old (random) tensor freed, new (checkpoint) tensor allocated
          on the same device — peak VRAM stays ≈ constant at ~14 GB.

        This replaces the old approach that loaded all 4 shards (~14 GB) into
        RAM simultaneously and then did a bulk .to(device="cuda") that OOM'd
        because the model already occupied 14 GB on the GPU.
        """
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for weight key remapping. "
                "Run: pip install safetensors"
            ) from exc

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(model_id, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot locate cached checkpoint for '{model_id}'. "
                f"Ensure the model has been downloaded. Error: {exc}"
            ) from exc

        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
        logger.info(
            "Streaming weights from %d shard(s) (in-place GPU replacement)…",
            len(shard_files),
        )

        # Build param/buffer metadata once before streaming starts.
        # Meta-device parameters arise when device_map="auto" dispatches a model
        # that has ALL keys MISSING (old checkpoint format): accelerate initialises
        # the skeleton on meta and never materialises it. We must pass a real
        # device to set_module_tensor_to_device so it allocates memory there.
        # Prefer the accelerate device map; fall back to cuda:0 / cpu.
        hf_device_map: dict = getattr(model, "hf_device_map", {})
        _cuda_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

        def _resolve_device(param_name: str, current: torch.device) -> str | torch.device:
            if current.type != "meta":
                return current  # already materialised; keep on its current device
            # Walk the accelerate device map to find the intended device for this param
            if hf_device_map:
                for module_path in sorted(hf_device_map, key=len, reverse=True):
                    if param_name == module_path or param_name.startswith(module_path + "."):
                        return hf_device_map[module_path]
            return _cuda_dev  # default: first GPU (or CPU on CPU-only machines)

        model_meta: dict[str, tuple[torch.dtype, str | torch.device]] = {}
        for name, param in model.named_parameters():
            model_meta[name] = (param.dtype, _resolve_device(name, param.device))
        for name, buf in model.named_buffers():
            model_meta[name] = (buf.dtype, _resolve_device(name, buf.device))

        # Prefer accelerate's set_module_tensor_to_device which handles meta→GPU
        # and all accelerate device-map hooks correctly.
        try:
            from accelerate.utils import set_module_tensor_to_device

            use_accelerate = True
        except ImportError:
            use_accelerate = False
            logger.warning(
                "accelerate not installed — falling back to direct parameter copy. "
                "Install accelerate for correct behaviour with device_map: "
                "pip install accelerate"
            )

        matched_count = 0
        unmatched: list[str] = []

        for shard_file in shard_files:
            shard_path = os.path.join(local_dir, shard_file)
            # safe_open with device="cpu" keeps the shard off the GPU.
            # Each tensor is released (del tensor) immediately after assignment
            # so the shard's memory is freed progressively within the loop.
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for old_key in f.keys():
                    new_key = _remap_key(old_key)
                    if new_key not in model_meta:
                        unmatched.append(old_key)
                        continue
                    dtype, target_device = model_meta[new_key]
                    tensor = f.get_tensor(old_key).to(dtype=dtype)
                    if use_accelerate:
                        set_module_tensor_to_device(
                            model, new_key, target_device, value=tensor
                        )
                    else:
                        _set_param_direct(model, new_key, tensor, target_device)
                    matched_count += 1
                    del tensor  # release immediately; don't accumulate in RAM
            # shard context exits → safe_open releases mmap → shard RAM freed

        if unmatched:
            logger.debug(
                "%d checkpoint keys had no model match (skipped): %s…",
                len(unmatched),
                unmatched[:3],
            )
        logger.info(
            "Weight remapping complete: %d/%d model params applied.",
            matched_count,
            len(model_meta),
        )

    def _load_processor(self, model_id: str):
        """Load the multimodal processor with fallbacks for all transformers versions.

        transformers 4.x: AutoProcessor resolves llava_mistral automatically.
        transformers 5.x: AutoProcessor requires processor_config.json which
            LLaVA-Med's repo does not ship (all 404). Falls back through two
            explicit strategies before giving up.

        Tier 1 — AutoProcessor.from_pretrained        (transformers 4.x)
        Tier 2 — LlavaProcessor.from_pretrained       (explicit class, no auto-detect)
        Tier 3 — Manual construction from known CLIP + tokenizer components
        """
        from transformers import AutoProcessor

        # Tier 1: standard auto-detect path
        try:
            return AutoProcessor.from_pretrained(model_id)
        except (ValueError, OSError) as e:
            logger.warning(
                "AutoProcessor.from_pretrained failed for '%s': %s. "
                "This is expected with transformers>=5.0 — the llava_mistral "
                "checkpoint has no processor_config.json. Trying LlavaProcessor...",
                model_id,
                e,
            )

        # Tier 2: explicit class — still reads config.json for image processor info
        try:
            from transformers import LlavaProcessor
            return LlavaProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.warning(
                "LlavaProcessor.from_pretrained also failed: %s. "
                "Building processor from known components for LLaVA-Med v1.5...",
                e,
            )

        # Tier 3: manual construction using the fixed architecture of LLaVA-Med v1.5.
        # Vision encoder: CLIP-ViT-L/14-336  (hard-coded in the model config)
        # Tokenizer:      Mistral-7B          (requires sentencepiece + protobuf)
        #
        # The fast tokenizer (use_fast=True, default) converts SentencePiece vocab
        # to the Rust tokenizers format using transformers's SentencePieceExtractor,
        # which requires the standalone `protobuf` package.  Without it, transformers
        # silently falls back to TikToken which crashes on binary .model files.
        # The slow tokenizer (use_fast=False) uses the sentencepiece Python binding
        # directly — its own bundled protobuf — so no standalone protobuf is needed.
        try:
            from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor

            _VISION_TOWER = "openai/clip-vit-large-patch14-336"
            logger.info(
                "Building LlavaProcessor from components: "
                "image_processor=%s, tokenizer=%s",
                _VISION_TOWER,
                model_id,
            )

            # Try fast tokenizer first (preferred: Rust-based, faster).
            # Falls back to slow tokenizer if protobuf is not installed.
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception as tok_err:
                logger.warning(
                    "Fast tokenizer failed (%s). "
                    "Retrying with use_fast=False (slow tokenizer, no protobuf needed). "
                    "Install protobuf to avoid this: pip install protobuf",
                    tok_err,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

            image_processor = CLIPImageProcessor.from_pretrained(_VISION_TOWER)
            return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception as e:
            raise RuntimeError(
                f"All three processor loading strategies failed for '{model_id}'. "
                f"Last error: {e}. "
                "Ensure sentencepiece and protobuf are installed: "
                "pip install sentencepiece protobuf"
            ) from e

    def _load_with_remapping(
        self,
        model_id: str,
        llava_cfg,
        torch_dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Load an old-format LLaVA-Med checkpoint without the AlignDevicesHook staleness bug.

        The bug in the previous approach
        ---------------------------------
        ``from_pretrained(device_map="auto")`` attaches ``AlignDevicesHook`` to
        CPU-offloaded layers and initialises each hook's ``weights_map`` from the
        parameter tensors **at dispatch time** (random, because all checkpoint keys
        were MISSING).  Subsequent calls to ``set_module_tensor_to_device`` update
        ``module._parameters`` but NOT ``hook.weights_map``.  On every forward pass
        ``pre_forward`` reloads the stale random tensors from ``weights_map``,
        overwriting the correctly-loaded checkpoint values → garbage output.

        Correct sequence: init → stream weights → dispatch
        ---------------------------------------------------
        1. ``init_empty_weights()``        — meta model, zero memory, no hooks
        2. ``infer_auto_device_map()``     — decide GPU/CPU placement
        3. ``set_module_tensor_to_device`` — materialise meta → real device
           (no hooks present yet → no staleness issue)
        4. ``dispatch_model()``            — attach hooks NOW, from already-correct weights
        """
        try:
            from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
        except ImportError as exc:
            raise ImportError(
                "accelerate is required for device_map='auto' weight loading. "
                "Install it with: pip install accelerate"
            ) from exc

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required for weight key remapping. "
                "Run: pip install safetensors"
            ) from exc

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        from transformers import LlavaForConditionalGeneration

        # ── Step 1: create model on meta device ──────────────────────────────
        with init_empty_weights():
            model = LlavaForConditionalGeneration(config=llava_cfg)

        # ── Step 2: infer device placement ───────────────────────────────────
        no_split = getattr(model, "_no_split_modules", [])
        if torch.cuda.is_available():
            # Leave ~1.5 GB headroom on the T4 (14.56 GB usable).
            # accelerate >= 0.27 requires integer keys for CUDA devices (not "cuda:0").
            max_mem: dict = {0: "13GiB", "cpu": "15GiB"}
        else:
            max_mem = {"cpu": "60GiB"}

        device_map: dict = infer_auto_device_map(
            model,
            max_memory=max_mem,
            dtype=torch_dtype,
            no_split_module_classes=no_split,
        )
        logger.info("device_map inferred: %d entries (GPU+CPU split).", len(device_map))

        # ── Step 3: build param → (dtype, target_device) lookup ──────────────
        # Walk named_parameters / named_buffers on the meta model to get names;
        # device comes from the device_map (not from param.device which is "meta").
        model_meta: dict[str, tuple[torch.dtype, str]] = {}
        for name, _ in itertools.chain(
            model.named_parameters(), model.named_buffers()
        ):
            target = "cpu"  # fallback: offload to CPU
            for module_path in sorted(device_map, key=len, reverse=True):
                if name == module_path or name.startswith(module_path + "."):
                    target = str(device_map[module_path])
                    break
            model_meta[name] = (torch_dtype, target)

        # ── Step 4: stream shards, materialise meta → target device ──────────
        local_dir = snapshot_download(model_id, local_files_only=True)
        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
        logger.info(
            "Streaming %d shard(s) → meta-to-device materialisation (no hooks yet)…",
            len(shard_files),
        )

        matched = 0
        unmatched: list[str] = []
        for shard_file in shard_files:
            shard_path = os.path.join(local_dir, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for old_key in f.keys():
                    new_key = _remap_key(old_key)
                    if new_key not in model_meta:
                        unmatched.append(old_key)
                        continue
                    dtype, target_device = model_meta[new_key]
                    tensor = f.get_tensor(old_key).to(dtype=dtype)
                    # No hooks attached yet → updates module._parameters directly,
                    # no weights_map staleness issue.
                    set_module_tensor_to_device(model, new_key, target_device, value=tensor)
                    matched += 1
                    del tensor  # release immediately
            # shard mmap released

        if unmatched:
            logger.debug(
                "%d checkpoint keys had no model match (skipped): %s…",
                len(unmatched),
                unmatched[:3],
            )
        logger.info(
            "Weight materialisation complete: %d/%d params loaded.",
            matched,
            len(model_meta),
        )

        # ── Step 5: dispatch NOW — hooks see already-correct weights ──────────
        model = dispatch_model(model, device_map=device_map)
        return model

    def _load_model(self, model_id: str):
        try:
            from transformers import LlavaForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers components: {e}. "
                "Ensure transformers>=4.37 is installed."
            ) from e

        processor = self._load_processor(model_id)

        # Check checkpoint key format before building kwargs so we can
        # downgrade quantization when key remapping is needed.
        old_format = self._detect_old_key_format(model_id)

        kwargs: dict = {}

        if old_format and self._quantize in ("4bit", "8bit"):
            # bitsandbytes quantizes weights in-place during from_pretrained.
            # That requires the checkpoint weights to map correctly.
            # With old-format keys (all MISSING → random init), bitsandbytes
            # creates LinearFP4 shells whose quant state is never initialised,
            # then asserts shape[1]==1 on the first forward pass and crashes.
            # Fix: skip bitsandbytes, load in fp16 on GPU instead.
            # fp16 7B ≈ 14 GB on T4 (fits within 16 GB VRAM).
            logger.warning(
                "bitsandbytes %s-bit quantization is incompatible with old-format "
                "LLaVA-Med checkpoints when using transformers >= 4.45 "
                "(all weights land as MISSING → random init → quant state "
                "never set → AssertionError on first forward pass). "
                "Loading in fp16 instead. "
                "To restore 4-bit quantization, pin the transformers version: "
                "  pip install 'transformers==4.44.2'",
                self._quantize,
            )
            kwargs["torch_dtype"] = torch.float16
            self._quantize = "fp16"  # reflect the actual precision in model_name

        elif self._quantize in ("4bit", "8bit") and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                if self._quantize == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                else:
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantization.")
                kwargs["torch_dtype"] = torch.float32

        else:
            kwargs["torch_dtype"] = torch.float32

        if self._device != "cpu" and torch.cuda.is_available():
            kwargs["device_map"] = self._device

        # ── Architecture fix for llava_mistral on transformers >= 5.x ────────
        # The LLaVA-Med config.json (written for transformers 4.36.x) is FLAT:
        # all Mistral-7B fields (hidden_size, intermediate_size=14336, …) sit at
        # the top level — there is no nested "text_config" sub-object.
        #
        # transformers 5.x does not recognise the "llava_mistral" model_type and
        # silently remaps it to the generic "llava" architecture BEFORE returning
        # from AutoConfig.from_pretrained.  This creates a default LlamaConfig
        # with intermediate_size=11008. Every FFN weight in the checkpoint
        # (14336) then mismatches the model skeleton (11008) and
        # set_module_tensor_to_device raises:
        #   ValueError: Trying to set a tensor of shape [4096,14336] in "weight"
        #               (which has shape [4096,11008])
        #
        # Fix: when old_format=True (already detected from shard index), read
        # config.json DIRECTLY as raw JSON (bypassing AutoConfig remapping),
        # build an explicit MistralConfig text sub-config, wrap it in LlavaConfig,
        # and pass as config= to from_pretrained.
        if old_format:
            try:
                from huggingface_hub import hf_hub_download
                from transformers import LlavaConfig as _LlavaConfig, MistralConfig as _MCfg

                # Read raw JSON — avoids transformers silently remapping
                # model_type="llava_mistral" → "llava" (LlamaConfig)
                _config_path = hf_hub_download(
                    repo_id=model_id, filename="config.json", local_files_only=True
                )
                with open(_config_path, encoding="utf-8") as _f:
                    _j = json.load(_f)

                # Build a typed MistralConfig so transformers 5.x uses the right
                # architecture class (intermediate_size=14336, not LLaMA's 11008).
                _mistral_cfg = _MCfg(
                    hidden_size=_j.get("hidden_size", 4096),
                    intermediate_size=_j.get("intermediate_size", 14336),
                    num_hidden_layers=_j.get("num_hidden_layers", 32),
                    num_attention_heads=_j.get("num_attention_heads", 32),
                    num_key_value_heads=_j.get("num_key_value_heads", 8),
                    max_position_embeddings=_j.get("max_position_embeddings", 32768),
                    vocab_size=_j.get("vocab_size", 32000),
                    rms_norm_eps=_j.get("rms_norm_eps", 1e-5),
                    rope_theta=_j.get("rope_theta", 1_000_000.0),
                    hidden_act=_j.get("hidden_act", "silu"),
                )

                # Vision config: load from cache or fall back to hardcoded values
                # for openai/clip-vit-large-patch14-336 (the hard-coded vision tower).
                _vis_tower = _j.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
                try:
                    from transformers import CLIPVisionConfig as _CVCfg

                    _vis_cfg = _CVCfg.from_pretrained(_vis_tower, local_files_only=True)
                except Exception:
                    _vis_cfg = None  # LlavaConfig will fill in defaults

                _llava_cfg = _LlavaConfig(
                    text_config=_mistral_cfg,
                    **({} if _vis_cfg is None else {"vision_config": _vis_cfg}),
                )
                # Map old-style mm_* fields to standard LlavaConfig fields
                _llava_cfg.vision_feature_layer = _j.get(
                    "mm_vision_select_layer", _j.get("vision_feature_layer", -2)
                )
                _llava_cfg.vision_feature_select_strategy = _j.get(
                    "vision_feature_select_strategy", "default"
                )
                for _attr in ("image_token_index", "projector_hidden_act"):
                    if _attr in _j:
                        setattr(_llava_cfg, _attr, _j[_attr])

                kwargs["config"] = _llava_cfg
                logger.info(
                    "Built explicit LlavaConfig (MistralConfig text_config) for "
                    "llava_mistral checkpoint: intermediate_size=%d "
                    "(prevents LLaMA-7B dimension fallback).",
                    _mistral_cfg.intermediate_size,
                )
            except Exception as _cfg_err:
                logger.debug("Config pre-processing skipped (%s).", _cfg_err)

        if old_format and torch.cuda.is_available():
            # Use init_empty_weights → stream → dispatch_model sequence to avoid
            # the AlignDevicesHook.weights_map staleness bug (see _load_with_remapping).
            torch_dtype = kwargs.get("torch_dtype", torch.float16)
            llava_cfg = kwargs.get("config")
            try:
                model = self._load_with_remapping(model_id, llava_cfg, torch_dtype)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load old-format model '{model_id}' via remapping path. "
                    f"Error: {e}"
                ) from e
        else:
            # New-format checkpoint, or CPU-only: standard from_pretrained path.
            try:
                model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
                if self._device == "cpu":
                    model = model.to("cpu")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{model_id}'. "
                    "If this is an OOM error, try quantize='4bit' or quantize='8bit' on GPU. "
                    f"Error: {e}"
                ) from e
            # Old-format on CPU: apply remapping after from_pretrained
            if old_format:
                self._apply_remapped_weights(model, model_id)

        model.eval()

        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info("GPU memory after model load: %.2f GB", vram_gb)

        return processor, model

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image, question: str) -> VLMPrediction:
        """Run inference on a single image-question pair."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = f"<image>\nUSER: {question}\nASSISTANT:"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self._inferred_device) for k, v in inputs.items()}

        start = time.perf_counter()
        # torch.inference_mode is strictly more efficient than no_grad for
        # pure inference: it additionally disables autograd version tracking.
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, input_len:]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True)
        answer = raw_output.strip() or "unknown"

        confidence = self._extract_confidence(output.scores, generated_ids)

        return VLMPrediction(
            answer=answer,
            confidence=confidence,
            raw_output=raw_output,
            model_name=self.model_name,
            latency_seconds=latency,
        )

    def _extract_confidence(
        self, scores: tuple, generated_ids: torch.Tensor
    ) -> float:
        """Compute mean token probability over the generated sequence.

        Replaces the original per-token Python loop with a single batched
        tensor operation:
          - stack scores:  (T, vocab_size)
          - softmax once:  (T, vocab_size)
          - gather token probs:  (T,)
          - mean → scalar
        Result: one GPU softmax + one gather instead of T sequential ops.
        """
        try:
            if not scores:
                return 0.5

            # scores: tuple of T tensors each (1, vocab_size)
            scores_tensor = torch.stack(scores).squeeze(1)  # (T, vocab_size)
            probs = torch.softmax(scores_tensor, dim=-1)     # (T, vocab_size)
            token_probs = probs[
                torch.arange(len(scores), device=probs.device),
                generated_ids[: len(scores)],
            ]  # (T,)
            return token_probs.mean().item()
        except Exception:
            return 0.5

    def predict_batch(
        self, samples: list[tuple[Image.Image, str]]
    ) -> list[VLMPrediction]:
        """Sequential batch inference.

        LLaVA-Med processes variable-length multimodal sequences; padding
        strategies for true batching are non-trivial and not yet implemented.
        Sequential predict() calls are used with consistent latency tracking.
        """
        return [self.predict(image, question) for image, question in samples]

    @property
    def model_name(self) -> str:
        return f"llava-med-v1.5-mistral-7b-{self._quantize}"
