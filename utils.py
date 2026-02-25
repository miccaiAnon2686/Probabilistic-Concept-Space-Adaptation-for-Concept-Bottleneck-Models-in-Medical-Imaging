import os
import math
import torch
import clip
import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


PM_SUFFIX = {"max":"_max", "avg":""}

def _is_no_clip(name):
    return name in {None, "", "none", "no-clip", "NO-CLIP", "No-Clip"}

# --- restore this in utils.py ---

from functools import reduce
import operator as _op

def _get_module_by_name(model, dotted_name: str):
    """
    Resolve submodule/attribute by dotted path, e.g. 'layer4.1.conv2' or 'conv_proj'.
    """
    return reduce(getattr, dotted_name.split("."), model)

def save_target_activations(target_model, dataset, save_name,
                            target_layers=("layer4",), batch_size=1000,
                            device="cuda", pool_mode="avg"):
    """
    Hook into arbitrary target layers, pool (avg|max) if 4D, and cache features to disk.
    `save_name` must contain a '{}' placeholder that will be formatted with the layer name.
    """
    # Build per-layer output paths
    save_paths = {layer: save_name.format(layer) for layer in target_layers}

    # If everything is already saved, bail early
    if _all_saved(save_paths):
        return

    # Make sure parent dirs exist
    for p in save_paths.values():
        _make_save_dir(p)

    # Storage for activations and hook handles
    all_feats = {layer: [] for layer in target_layers}
    hooks = {}

    # Register hooks
    for layer in target_layers:
        mod = _get_module_by_name(target_model, layer)
        hooks[layer] = mod.register_forward_hook(get_activation(all_feats[layer], pool_mode))

    # Run forward passes to trigger hooks
    target_model.eval()
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            # Support (x,y) or (x,y,...) or just x
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            _ = target_model(images.to(device))

    # Save and clean up
    for layer in target_layers:
        torch.save(torch.cat(all_feats[layer], dim=0), save_paths[layer])
        hooks[layer].remove()

    # free memory
    del all_feats
    torch.cuda.empty_cache()


# def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
#                             device = "cuda", pool_mode='avg'):
#     """
#     save_name: save_file path, should include {} which will be formatted by layer names
#     """
#     _make_save_dir(save_name)
#     save_names = {}
#     for target_layer in target_layers:
#         save_names[target_layer] = save_name.format(target_layer)

#     if _all_saved(save_names):
#         return

#     all_features = {target_layer:[] for target_layer in target_layers}

#     hooks = {}
#     for target_layer in target_layers:
#         command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
#         hooks[target_layer] = eval(command)

#     with torch.no_grad():
#         # for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
#         for batch in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
#             # allow (x,y) or (x,y,...) tuples
#             if isinstance(batch, (list, tuple)) and len(batch) >= 2:
#                 images, labels = batch[0], batch[1]
#             else:
#                 images, labels = batch  # keep old behavior for 2-tuples
#             _ = target_model(images.to(device))  # forward to trigger hooks

#     for target_layer in target_layers:
#         torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
#         hooks[target_layer].remove()
#     #free memory
#     del all_features
#     torch.cuda.empty_cache()
#     return

def _get_x_from_batch(batch):
    # Accept tuple/list/dict or raw tensor; return images tensor on device
    if isinstance(batch, (list, tuple)):
        xb = batch[0]
    elif isinstance(batch, dict):
        xb = batch.get("image", next(iter(batch.values())))
    else:
        xb = batch
    return xb

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    """
    Compute & cache:
      - Target backbone activations (hooked or ViT-CLS preproj) to target_save_name.format(layer)
      - If clip_name provided:
          * For OpenAI CLIP: concept text feats + image feats
          * For hf-hub (open_clip): concept text feats + image feats + ENSEMBLED CLASS text feats
    """

    # ---------------------------
    # Helpers (scoped to function)
    # ---------------------------
    def _build_biomed_class_prompts(class_names):
        PROMPT_TEMPLATES = [
            "a light microscopy image of a {cls} white blood cell on a peripheral blood smear",
            "a Wright–Giemsa stained blood smear showing a {cls} leukocyte",
            "a cytology image: {cls} leukocyte under brightfield microscopy",
            "a hematology slide with a {cls} cell",
            "a peripheral blood smear image of a {cls} cell",
        ]
        BIOMED_WBC_SYNONYMS = {
            "Basophil":    ["basophil", "basophilic granulocyte"],
            "Eosinophil":  ["eosinophil", "eosinophilic granulocyte"],
            "Lymphocyte":  ["lymphocyte", "small lymphocyte"],
            "Monocyte":    ["monocyte"],
            "Neutrophil":  ["neutrophil", "segmented neutrophil", "band neutrophil", "polymorphonuclear neutrophil"],
        }
        prompts, spans = [], []
        for cname in class_names:
            synonyms = BIOMED_WBC_SYNONYMS.get(cname, [cname])
            start = len(prompts)
            for syn in synonyms:
                for tmpl in PROMPT_TEMPLATES:
                    prompts.append(tmpl.format(cls=syn))
            spans.append((start, len(prompts)))
        return prompts, spans

    @torch.no_grad()
    def _encode_class_text_openclip(model, tokenizer, class_names, save_name, device="cuda", batch_size=256):
        if os.path.exists(save_name):
            return
        prompts, spans = _build_biomed_class_prompts(class_names)

        toks_all = []
        for i in range(0, len(prompts), batch_size):
            toks_all.append(tokenizer(prompts[i:i+batch_size]))
        toks = torch.cat(toks_all, dim=0).to(device)

        feats = []
        for i in range(0, toks.size(0), batch_size):
            f = model.encode_text(toks[i:i+batch_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0).float()  # [S, d]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        K = len(class_names); d = feats.size(1)
        class_feats = torch.empty(K, d, device=feats.device, dtype=feats.dtype)
        for k, (s, e) in enumerate(spans):
            class_feats[k] = feats[s:e].mean(dim=0)
        torch.save(class_feats.cpu(), save_name)

    @torch.no_grad()
    def _encode_concept_text_openclip(model, tokenizer, words, save_name, device="cuda", batch_size=256):
        if os.path.exists(save_name):
            return
        toks_all = []
        for i in range(0, len(words), batch_size):
            toks_all.append(tokenizer(words[i:i+batch_size]))
        toks = torch.cat(toks_all, dim=0).to(device)

        feats = []
        for i in range(0, toks.size(0), batch_size):
            f = model.encode_text(toks[i:i+batch_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0).float()
        torch.save(feats.cpu(), save_name)

    @torch.no_grad()
    def _encode_images_openclip(model, dataset, save_name, batch_size=1000, device="cuda"):
        if os.path.exists(save_name):
            return
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        all_features = []
        for batch in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch
            feats = model.encode_image(images.to(device))
            all_features.append(feats.detach().cpu())
        torch.save(torch.cat(all_features, dim=0), save_name)

    # ---------------------------
    # Prep save names & early-exit
    # ---------------------------
    use_clip = not _is_no_clip(clip_name)
    target_save_name, clip_save_name, text_save_name = get_save_names(
        clip_name, target_name, "{}", d_probe, concept_set, pool_mode, save_dir
    )

    # Build required files dict for early bail-out (do NOT force class-text file)
    save_names = {layer: target_save_name.format(layer) for layer in target_layers}
    if use_clip:
        save_names["clip"] = clip_save_name
        save_names["text"] = text_save_name
    if _all_saved(save_names):
        return

    # ---------------------------
    # Target backbone features
    # ---------------------------
    # (1) Build target model + dataset
    if target_name.startswith("clip_"):
        import clip as oai_clip
        target_model, target_preprocess = oai_clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = get_target_model(target_name, device)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    # (2) Save target activations or CLIP image features
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)

    # ---------------------------
    # CLIP branch (optional)
    # ---------------------------
    if use_clip:
        # Distinguish OpenAI CLIP vs open_clip (hf-hub)
        is_openclip = isinstance(clip_name, str) and clip_name.startswith("hf-hub:")

        if is_openclip:
            # ---- open_clip path (BiomedCLIP etc.) ----
            import open_clip
            model_id = clip_name[len("hf-hub:"):]
            oc_model, oc_preprocess_train, oc_preprocess_val = open_clip.create_model_and_transforms(model_id, device=device)
            oc_model.eval()
            oc_preprocess = oc_preprocess_val or oc_preprocess_train
            data_c = data_utils.get_data(d_probe, oc_preprocess)

            # Save image feats with open_clip
            _encode_images_openclip(oc_model, data_c, clip_save_name, batch_size=batch_size, device=device)

            # Concept text features from concept_set (list of words/phrases)
            with open(concept_set, 'r') as f:
                words = [w for w in f.read().split('\n') if len(w.strip()) > 0]
            tokenizer = open_clip.get_tokenizer(model_id)
            _encode_concept_text_openclip(oc_model, tokenizer, words, text_save_name, device=device, batch_size=batch_size)

            # ENSEMBLED CLASS text features (domain prompts + synonyms)
            #   Path your adapter already prefers:
            clip_tag = ("" if _is_no_clip(clip_name) else clip_name).replace('/', '')
            class_text_save_name = f"{save_dir}/{d_probe}_classes_photo_{clip_tag}.pt"

            # Prefer dataset-provided class names; fallback to default WBC 5-class list
            get_cls = getattr(data_utils, "get_class_names", None)
            if callable(get_cls):
                class_names = get_cls(d_probe)
            else:
                class_names = ["Basophil","Eosinophil","Lymphocyte","Monocyte","Neutrophil"]

            _encode_class_text_openclip(oc_model, tokenizer, class_names, class_text_save_name, device=device, batch_size=batch_size)

        else:
            # ---- OpenAI CLIP path (your original behavior) ----
            import clip as oai_clip
            clip_model, clip_preprocess = oai_clip.load(clip_name, device=device)
            data_c = data_utils.get_data(d_probe, clip_preprocess)

            # Concept text (from concept_set) via OpenAI CLIP tokenizer/encoder
            with open(concept_set, 'r') as f: 
                words = (f.read()).split('\n')
            text = oai_clip.tokenize([f"{w}" for w in words]).to(device)

            save_clip_text_features(clip_model, text, text_save_name, batch_size)
            save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)

    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000, device="cuda", get_x=None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return

    # fallback extractor if none provided
    if get_x is None:
        def get_x(batch):
            if isinstance(batch, (list, tuple)):
                return batch[0]
            elif isinstance(batch, dict):
                # try common keys; otherwise first value
                return batch.get("image", next(iter(batch.values())))
            return batch

    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            images = get_x(batch)
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text_tokens, save_name, batch_size=1000):
    """
    Works for:
      - OpenAI CLIP: text_tokens is a [N, context] LongTensor
      - OpenCLIP/HF: text_tokens is a dict of Tensors with matching first dim
    """
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []

    def _num_items(tok):
        if isinstance(tok, dict):
            # infer length from one key
            key = next(iter(tok))
            return tok[key].shape[0]
        return len(tok)

    def _slice(tok, s, e):
        if isinstance(tok, dict):
            return {k: v[s:e] for k, v in tok.items()}
        return tok[s:e]

    N = _num_items(text_tokens)
    with torch.no_grad():
        for i in tqdm(range(math.ceil(N / batch_size))):
            s, e = i * batch_size, min((i + 1) * batch_size, N)
            batch = _slice(text_tokens, s, e)
            feats = model.encode_text(batch)
            text_features.append(feats.cpu())

    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()

def _resolve_openclip_hub_id(clip_name: str) -> str:
    name = (clip_name or "").strip()
    if name.startswith("hf-hub:"):
        return name
    low = name.lower().replace("/", "_").replace("-", "_")
    if "biomedclip" in low:
        return "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    return name


def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    """
    Cache:
      • Target/backbone features (hook-based or CLIP visual 'encode_image')
      • (If using CLIP) CLIP image features
      • (If using CLIP) CLIP text features for:
            - concept prompts (from concept_set file)  [existing]
            - class prompts  (from dataset class names) [new, if available]
    Supports:
      • OpenAI CLIP via `clip.load(...)` (names like 'ViT-B/16')
      • OpenCLIP/HF via `open_clip.create_model_from_pretrained("hf-hub:<repo>")`
        e.g., 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    """

    def _is_no_clip(name):
        return name in {None, "", "none", "no-clip", "NO-CLIP", "No-Clip"}

    def _make_save_dir(path):
        d = path[: path.rfind("/")]
        if not os.path.exists(d):
            os.makedirs(d)

    def _all_saved(paths_dict):
        for p in paths_dict.values():
            if not os.path.exists(p):
                return False
        return True

    # helper: encode and save text features for both OpenAI CLIP and OpenCLIP/HF tokenizers
    def _encode_and_save_text_feats_texttokens(model, text_tokens, save_path, bs=1000):
        """
        text_tokens can be:
          - Tensor [N, context]           (OpenAI CLIP)
          - dict of Tensors (e.g., HF)    (OpenCLIP)
        """
        if os.path.exists(save_path):
            return
        _make_save_dir(save_path)

        def _num_items(tok):
            if isinstance(tok, dict):
                return next(iter(tok.values())).shape[0]
            return tok.shape[0]

        def _slice(tok, s, e):
            if isinstance(tok, dict):
                return {k: v[s:e] for k, v in tok.items()}
            return tok[s:e]

        feats_all = []
        N = _num_items(text_tokens)
        with torch.no_grad():
            for i in tqdm(range(math.ceil(N / bs))):
                s, e = i * bs, min((i + 1) * bs, N)
                batch_tok = _slice(text_tokens, s, e)
                feats = model.encode_text(batch_tok)
                feats_all.append(feats.cpu())
        feats_all = torch.cat(feats_all, dim=0)
        torch.save(feats_all, save_path)
        del feats_all
        torch.cuda.empty_cache()

    # Build core save names via your helper
    target_save_name, clip_save_name, text_save_name = get_save_names(
        clip_name, target_name, "{}", d_probe, concept_set, pool_mode, save_dir
    )

    # We'll also optionally save class-prompt text feats in:
    #   <save_dir>/<d_probe>_classes_photo_<cliptag>.pt
    clip_tag = "" if _is_no_clip(clip_name) else clip_name
    class_text_save_name = "{}/{}_classes_{}_{}.pt".format(
        save_dir, d_probe, "photo", ("" if clip_tag is None else clip_tag).replace("/", "")
    )

    # Aggregate "do we already have everything?"
    save_names = {}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)
    use_clip = not _is_no_clip(clip_name)
    if use_clip:
        save_names["clip_image"] = clip_save_name
        save_names["clip_text_concepts"] = text_save_name
        # class prompts are optional; include in all-saved check only if class names exist
        try:
            _cls = data_utils.get_class_names(d_probe)
            if _cls and len(_cls) > 0:
                save_names["clip_text_classes"] = class_text_save_name
        except Exception:
            pass

    # Short-circuit if everything exists
    if _all_saved(save_names):
        return

    # -------------------------
    # Load target model + data
    # -------------------------
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    # -------------------------
    # Optional CLIP branch
    # -------------------------
    if use_clip:
        clip_model = None
        clip_preprocess = None
        tokenizer = None
        is_openai_clip = False

        # Try OpenAI CLIP first
        try:
            clip_model, clip_preprocess = clip.load(clip_name, device=device)
            tokenizer = clip.tokenize
            is_openai_clip = True
            clip_model = clip_model.to(device).eval()   # <-- ADD THIS
        except Exception:
            # Fallback to OpenCLIP/HF
            try:
                import open_clip
            except Exception as e:
                raise RuntimeError(
                    f"Requested CLIP '{clip_name}' but open_clip_torch is not installed. "
                    f"Install with: pip install open_clip_torch transformers huggingface_hub"
                ) from e

            hub_id = (clip_name or "").strip()
            low = hub_id.lower().replace("/", "_").replace("-", "_")
            if not hub_id.startswith("hf-hub:") and "biomedclip" in low:
                hub_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

            if hub_id.startswith("hf-hub:"):
                clip_model, clip_preprocess = open_clip.create_model_from_pretrained(hub_id)
                tokenizer = open_clip.get_tokenizer(hub_id)
            else:
                clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained=hub_id
                )
                tokenizer = open_clip.get_tokenizer("ViT-B-16")

            clip_model = clip_model.to(device).eval()   # <-- ADD THIS

        # Build dataset for CLIP image feats
        data_c = data_utils.get_data(d_probe, clip_preprocess)

        # ---- (1) Concept text feats (existing behavior) ----
        with open(concept_set, "r") as f:
            words = [w for w in (f.read()).split("\n") if len(w.strip()) > 0]

        if is_openai_clip:
            text_tokens = tokenizer([f"{w}" for w in words]).to(device)
        else:
            # OpenCLIP/HF tokenizers often return a dict of tensors
            text_tokens = tokenizer([f"{w}" for w in words], context_length=256)
            if isinstance(text_tokens, dict):
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            else:
                text_tokens = text_tokens.to(device)

        _encode_and_save_text_feats_texttokens(
            clip_model, text_tokens, text_save_name, batch_size
        )

        # ---- (2) Class-name text feats (NEW, optional) ----
        try:
            class_names = data_utils.get_class_names(d_probe)
        except Exception:
            class_names = None

        if class_names and len(class_names) > 0:
            prompts = [f"a photo of a {c}" for c in class_names]
            if is_openai_clip:
                class_tokens = tokenizer(prompts).to(device)
            else:
                class_tokens = tokenizer(prompts, context_length=256)
                if isinstance(class_tokens, dict):
                    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}
                else:
                    class_tokens = class_tokens.to(device)

            _encode_and_save_text_feats_texttokens(
                clip_model, class_tokens, class_text_save_name, batch_size
            )

        # ---- (3) CLIP image feats (same call for OpenAI & OpenCLIP) ----
        save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)

    # -------------------------
    # Target activations/features
    # -------------------------

    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device, get_x=_get_x_from_batch)
    else:
        # Torchvision ViT returns CLS from forward() (after heads=Identity)
        try:
            from torchvision.models.vision_transformer import VisionTransformer
            is_tv_vit = isinstance(target_model, VisionTransformer)
        except Exception:
            is_tv_vit = False

        if is_tv_vit:
            # import torch
            from torch.utils.data import DataLoader
            target_model = target_model.to(device).eval()
            # ensure head removed (if your factory didn’t)
            try:
                import torch.nn as nn
                if hasattr(target_model, "heads"):
                    target_model.heads = nn.Identity()
            except Exception:
                pass

            save_path = target_save_name.format("cls")
            feats_all = []
            loader = DataLoader(data_t, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            with torch.no_grad():
                for batch in loader:
                    xb = _get_x_from_batch(batch).to(device, non_blocking=True)
                    f = target_model(xb)           # [B, 768]
                    if f.dim() == 3:               # ultra-defensive: take CLS if a sequence is returned
                        f = f[:, 0]
                    feats_all.append(f.detach().cpu())
            feats = torch.cat(feats_all, 0)
            _make_save_dir(save_path)
            torch.save(feats, save_path)
        else:
            # your existing generic hook-based path
            save_target_activations(
                target_model, data_t, target_save_name, target_layers,
                batch_size, device, pool_mode, get_x=_get_x_from_batch
            )

    return

    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

def _sanitize_tag(s: str) -> str:
    return (s or "").replace("/", "").replace("\\", "")

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    target_tag = _sanitize_tag(target_name)
    clip_tag = "" if _is_no_clip(clip_name) else _sanitize_tag(clip_name)

    if target_name.startswith("clip_"):
        target_save_name = f"{save_dir}/{d_probe}_{target_tag}.pt"
    else:
        target_save_name = f"{save_dir}/{d_probe}_{target_tag}_{target_layer}{PM_SUFFIX[pool_mode]}.pt"

    clip_save_name = f"{save_dir}/{d_probe}_clip_{clip_tag}.pt"
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = f"{save_dir}/{concept_set_name}_{clip_tag}.pt"
    return target_save_name, clip_save_name, text_save_name


# --- Add this in utils.py ---

def get_class_text_save_name(clip_name, d_probe, save_dir, prompt_tag="photo"):
    """
    Returns a path for class-name prompt embeddings, e.g.:
      <save_dir>/<d_probe>_classes_<prompt_tag>_<cliptag>.pt
    """
    clip_tag = "" if _is_no_clip(clip_name) else clip_name
    fname = f"{d_probe}_classes_{prompt_tag}_{('' if clip_tag is None else clip_tag).replace('/', '')}.pt"
    return os.path.join(save_dir, fname)

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred
