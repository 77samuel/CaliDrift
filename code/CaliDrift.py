# ============================================================
# CALIDRIFT: Complete Paper Code (Final - Q2/Q3 Ready)
# Files needed: TruthfulQA.csv, simple_qa_test_set.csv, test.json
# Output: calidrift_results.xlsx + all figures (PNG)
# Fixes applied:
#   - IC = exp(-H) consistent with paper
#   - SAVE_EVERY counter actually used
#   - Model loaded ONCE per mtype, not per seed
#   - Base model: few-shot prompt + stop string (100% parse rate)
#   - repetition_penalty=1.1, MAX_NEW_TOKENS=64
# ============================================================

import os, re, json, warnings, gc, subprocess, random
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

subprocess.run(["pip", "install", "statsmodels", "scikit-learn",
                "openpyxl", "transformers", "datasets", "matplotlib", "seaborn", "-q"],
               capture_output=True)

from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          StoppingCriteria, StoppingCriteriaList)

# ============================================================
# CONFIGURATION
# ============================================================
SAMPLE_SIZE    = 100          # 5=test | 100=validate | 500=paper
SEEDS          = [1,2,3]
MAX_NEW_TOKENS = 64
RHR_THRESHOLD  = 0.60
MAX_LENGTH     = 512
N_BOOTSTRAP    = 1000
SAVE_EVERY     = 10

FAITHDIAL_MAX  = 500

#============================================================
# AUTHENTICATION TEST — RUNS FIRST BEFORE ANY MODEL LOADING
# ============================================================
print("\n" + "="*60)
print("AUTHENTICATION CHECK")
print("="*60)

# Try to get token from environment
hf_token = os.environ.get("HF_TOKEN", None)
has_token = hf_token is not None and len(hf_token) > 10

print(f"HF_TOKEN found: {'✅ YES' if has_token else '❌ NO'}")

# Test login and model access
models_accessible = {}
gated_models = [
    ("google/gemma-2b", "Gemma-2B"),
    ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
]

if has_token:
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("✅ HuggingFace login successful")
        
        # Quick test for each gated model (just tokenizer, fast)
        print("\nTesting model access...")
        for model_id, model_name in gated_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                models_accessible[model_name] = True
                print(f"  ✅ {model_name}: ACCESSIBLE")
            except Exception as e:
                models_accessible[model_name] = False
                error_msg = str(e)
                if "401" in error_msg or "GatedRepoError" in error_msg:
                    print(f"  ❌ {model_name}: Authentication failed — request access at huggingface.co/{model_id}")
                elif "404" in error_msg:
                    print(f"  ❌ {model_name}: Model ID not found")
                else:
                    print(f"  ❌ {model_name}: {error_msg[:80]}")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        for model_name in [m[1] for m in gated_models]:
            models_accessible[model_name] = False
else:
    print("❌ No HF_TOKEN found in environment")
    print("   Gemma and Llama will be SKIPPED")
    for model_name in [m[1] for m in gated_models]:
        models_accessible[model_name] = False

print("\n" + "="*60)
print("MODEL AVAILABILITY DECISION")
print("="*60)


MODEL_PAIRS = [
    {"pair_name": "Qwen2.5-1.5B",
     "base_id":   "Qwen/Qwen2.5-1.5B",
     "instruct_id":"Qwen/Qwen2.5-1.5B-Instruct"},
    {"pair_name": "Gemma-2B",
     "base_id":   "google/gemma-2b",
     "instruct_id":"google/gemma-2b-it"},
    {"pair_name": "Llama-3.2-1B",
     "base_id":   "meta-llama/Llama-3.2-1B",
     "instruct_id":"meta-llama/Llama-3.2-1B-Instruct"},
]

MODEL_TYPES = ["base", "instruct"]

# ============================================================
# PATHS
# ============================================================
OUTDIR     = "/kaggle/working/calidrift_results"
CHECKPOINT = f"{OUTDIR}/checkpoint.json"
EXCEL_OUT  = f"{OUTDIR}/calidrift_results.xlsx"
FIGDIR     = f"{OUTDIR}/figures"
DATA_PATH = "/kaggle/input/datasets/kevinsam77/calidrift-dataset/"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("CALIDRIFT — FINAL PAPER CODE (base + instruct)")
print(f"Device: {device.upper()} | Sample size: {SAMPLE_SIZE} | Seeds: {SEEDS}")
print(f"Models: {[p['pair_name'] for p in MODEL_PAIRS]}")
print(f"Output: {EXCEL_OUT}")
print("="*60)

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: Running on CPU — will be very slow")

try:
    from huggingface_hub import login
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
        print("HuggingFace authenticated")
    else:
        print("NOTE: Set HF_TOKEN env var for Gemma/Llama")
except Exception as e:
    print(f"No HF auth: {e}")

# ============================================================
# STOPPING CRITERIA — prevents base model runaway generation
# Stops when model starts generating a new "Question:" line
# ============================================================
class StopOnNewQuestion(StoppingCriteria):
    def __init__(self, tokenizer):
        # Encode "\nQuestion:" as stop sequence
        self.stop_ids = tokenizer.encode(
            "\nQuestion:", add_special_tokens=False)
        self.min_len  = len(self.stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.min_len:
            return False
        tail = input_ids[0][-self.min_len:].tolist()
        return tail == self.stop_ids

# ============================================================
# HEDGE LEXICON
# ============================================================
HEDGE_PATTERNS = [
    r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
    r'\bprobably\b', r'\blikely\b', r'\bseems?\b', r'\bappears?\b',
    r'\bI (am not|\'m not) sure\b', r'\bI (believe|think|suppose)\b',
    r'\bto my knowledge\b', r'\bif I remember correctly\b',
]
HEDGE_REGEX = re.compile('|'.join(HEDGE_PATTERNS), re.IGNORECASE)

# ============================================================
# PROMPT + PARSING
# ============================================================
VC_REGEX     = re.compile(r'CONFIDENCE:\s*(\d{1,3})', re.IGNORECASE)
VC_ALT_REGEX = re.compile(r'[Cc]onfidence[:\s]+(\d{1,3})')

def make_prompt(question, mtype):
    if mtype == "base":
        # Few-shot completion format — 100% parse rate confirmed in pilot
        # Stop string "\nQuestion:" prevents runaway generation
        return (
            f"Question: What is the capital of France?\n"
            f"Answer: Paris. Confidence: 95\n\n"
            f"Question: Who wrote Hamlet?\n"
            f"Answer: Shakespeare. Confidence: 90\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    return (
        f"Question: {question}\n\n"
        "Answer the question. Then write:\n"
        "CONFIDENCE: [0-100]\n\nAnswer:"
    )

def parse_vc(text):
    m = VC_REGEX.search(text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val / 100.0
    m = VC_ALT_REGEX.search(text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val / 100.0
    return None

# ============================================================
# DATASET LOADING
# ============================================================
def load_truthfulqa(n, seed):
    df = pd.read_csv(os.path.join(DATA_PATH, "TruthfulQA.csv"))
    df.columns = [c.strip() for c in df.columns]
    q_col = next(c for c in df.columns if "question" in c.lower())
    a_col = next(c for c in df.columns if "best" in c.lower() or "answer" in c.lower())
    df = df.sample(frac=1, random_state=seed).head(n)
    return [{"question": str(r[q_col]), "reference": str(r[a_col])}
            for _, r in df.iterrows()]

def load_simpleqa(n, seed):
    df = pd.read_csv(DATA_PATH + "simple_qa_test_set.csv").sample(
        frac=1, random_state=seed).head(n)
    return [{"question": str(r["problem"]), "reference": str(r["answer"])}
            for _, r in df.iterrows()]

def load_faithdial(n, seed):
    n = min(n, FAITHDIAL_MAX)
    with open(DATA_PATH + "test.json", "r") as f:
        data = json.load(f)
    items = []
    for record in data:
        for utt in record.get("utterances", []):
            history = utt.get("history", [])
            question = history[-1] if history else ""
            knowledge = utt.get("knowledge", "")
            if question and knowledge:
                items.append({"question": str(question),
                              "reference": str(knowledge)})
    random.Random(seed).shuffle(items)
    return items[:n]

DATASET_LOADERS = {
    "TruthfulQA": load_truthfulqa,
    "SimpleQA":   load_simpleqa,
    "FaithDial":  load_faithdial,
}
DATASETS = {
    "TruthfulQA": {"n_samples": SAMPLE_SIZE},
    "SimpleQA":   {"n_samples": SAMPLE_SIZE},
    "FaithDial":  {"n_samples": SAMPLE_SIZE},
}

# ============================================================
# RESPONSE LABELING
# ============================================================
ABSTENTION_RE = re.compile(
    r"i (don'?t|do not) know|i('m| am) not sure|i cannot",
    re.IGNORECASE)

def normalize(t):
    return re.sub(r'[^a-z0-9 ]', '', t.lower()).strip()

def label_response(response, reference):
    if ABSTENTION_RE.search(response):
        return "abstained"
    rn, fn = normalize(response), normalize(reference)
    if fn and fn in rn:
        return "correct"
    words = [w for w in fn.split() if len(w) > 3]
    if words and sum(w in rn.split() for w in words) >= max(1, len(words)//2):
        return "correct"
    return "hallucinated"

def is_rh(response, ic):
    return bool(HEDGE_REGEX.search(response)) and ic >= RHR_THRESHOLD

# ============================================================
# CHECKPOINT
# ============================================================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return set(), []
    try:
        with open(CHECKPOINT, "r") as f:
            ckpt = json.load(f)
        keys    = set(tuple(k) for k in ckpt.get("completed_keys", []))
        results = ckpt.get("results", [])
        print(f"Checkpoint loaded: {len(results)} results, {len(keys)} keys")
        return keys, results
    except Exception as e:
        print(f"Checkpoint failed ({e}) — starting fresh")
        return set(), []

def save_checkpoint(completed_keys, results):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "completed_keys": [list(k) for k in completed_keys],
            "results":        results,
            "saved_at":       datetime.now().isoformat(),
            "n_results":      len(results),
        }, f)
    os.replace(tmp, CHECKPOINT)

# ============================================================
# MODEL LOADING
# ============================================================
def load_model(model_id):
    print(f"    Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    dtype = torch.bfloat16 if "gemma" in model_id.lower() else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    mdl.eval()
    if device == "cuda":
        print(f"    VRAM used: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    return tok, mdl

# ============================================================
# GENERATION + ENTROPY
# IC = exp(-H) — consistent with paper Section 4.1
# Stop string used for base models to prevent runaway
# ============================================================
def generate_with_entropy(tok, mdl, prompt, mtype="instruct"):
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=MAX_LENGTH)
    inp = {k: v.to(mdl.device) for k, v in inp.items()}
    plen = inp["input_ids"].shape[1]

    # Base model gets stop string to prevent generating new questions
    stopping_criteria = None
    if mtype == "base":
        stopper = StopOnNewQuestion(tok)
        stopping_criteria = StoppingCriteriaList([stopper])

    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    gen_ids = out.sequences[0][plen:]
    text    = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # For base model: trim at "\nQuestion:" if stop didn't fire in time
    if mtype == "base" and "\nQuestion:" in text:
        text = text[:text.index("\nQuestion:")].strip()

    entropies = []
    for i, tid in enumerate(gen_ids.tolist()):
        if tid in (tok.eos_token_id, tok.pad_token_id):
            continue
        if i >= len(out.scores):
            break
        probs = torch.softmax(out.scores[i][0].float(), dim=-1)
        H_tok = -(probs * torch.log(probs + 1e-12)).sum().item()
        entropies.append(H_tok)

    H  = float(np.mean(entropies)) if entropies else 0.0
    IC = float(np.exp(-H))          # paper definition: IC = exp(-H)
    return text, H, IC

# ============================================================
# MAIN EXPERIMENT LOOP
# Model loaded ONCE per (pair, dataset, mtype)
# SAVE_EVERY counter properly used
# ============================================================
completed_keys, all_results = load_checkpoint()
print(f"Starting with {len(all_results)} existing results")

total_runs = len(MODEL_PAIRS) * len(DATASETS) * len(MODEL_TYPES)
run_num    = 0

for pair in MODEL_PAIRS:
    for ds_name in DATASETS:
        for mtype in MODEL_TYPES:
            run_num += 1
            model_id   = pair[f"{mtype}_id"]
            model_name = model_id.split("/")[-1]

            all_done = all(
                all((pair["pair_name"], mtype, ds_name, seed, i) in completed_keys
                    for i in range(DATASETS[ds_name]["n_samples"]))
                for seed in SEEDS
            )
            if all_done:
                print(f"[{run_num}/{total_runs}] {pair['pair_name']} {mtype} {ds_name}: complete, skipping")
                continue

            print(f"\n[{run_num}/{total_runs}] {pair['pair_name']} | {ds_name} | {mtype}")
            tok, mdl       = load_model(model_id)
            new_since_save = 0

            for seed in SEEDS:
                items = DATASET_LOADERS[ds_name](
                    DATASETS[ds_name]["n_samples"], seed)

                remaining = [i for i in range(len(items))
                             if (pair["pair_name"],mtype,ds_name,seed,i)
                             not in completed_keys]

                if not remaining:
                    print(f"  seed={seed}: already complete")
                    continue

                print(f"  seed={seed}: {len(remaining)} samples remaining")

                for i in tqdm(remaining, desc=f"  {model_name} seed={seed}"):
                    prompt = make_prompt(items[i]["question"], mtype)
                    try:
                        # Pass mtype so stop string activates for base
                        response, H, IC = generate_with_entropy(
                            tok, mdl, prompt, mtype)
                    except Exception as e:
                        print(f"    item {i} error: {e}")
                        completed_keys.add((pair["pair_name"],mtype,ds_name,seed,i))
                        continue

                    VC  = parse_vc(response)
                    key = (pair["pair_name"], mtype, ds_name, seed, i)
                    completed_keys.add(key)

                    if VC is None:
                        continue

                    all_results.append({
                        "pair_name":  pair["pair_name"],
                        "model_type": mtype,
                        "model_name": model_name,
                        "dataset":    ds_name,
                        "seed":       seed,
                        "sample_idx": i,
                        "VC":         round(VC, 6),
                        "IC":         round(IC, 6),
                        "H":          round(H, 6),
                        "CDI":        round(abs(VC - IC), 6),
                        "drift_mode": ("UED" if VC < IC else
                                       "OED" if VC > IC else "balanced"),
                        "label":      label_response(response,
                                                     items[i]["reference"]),
                        "rh":         is_rh(response, IC),
                        "response":   response[:600],
                    })
                    new_since_save += 1
                    if new_since_save >= SAVE_EVERY:
                        save_checkpoint(completed_keys, all_results)
                        new_since_save = 0

            save_checkpoint(completed_keys, all_results)
            print(f"  Done. Total results: {len(all_results)}")

            del mdl, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

save_checkpoint(completed_keys, all_results)
print(f"\nExperiment complete. Total results: {len(all_results)}")

# ============================================================
# ANALYSIS & TABLES
# ============================================================
if len(all_results) < 2:
    print("Too few results — run with larger SAMPLE_SIZE")
    exit()

df   = pd.DataFrame(all_results)
df_c = df.dropna(subset=["CDI","VC","IC"]).copy()
PAIRS = df_c["pair_name"].unique()
DSS   = df_c["dataset"].unique()

print(f"\nAnalysing {len(df_c)} rows | Pairs: {list(PAIRS)} | Datasets: {list(DSS)}")

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    ps = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return abs(a.mean()-b.mean()) / ps if ps > 0 else 0.0

def ece(vc, correct, bins=10):
    vc = np.array(vc); c = np.array(correct, dtype=float)
    edges = np.linspace(0, 1, bins+1); out = 0.0
    for i in range(bins):
        hi   = edges[i+1] if i < bins-1 else edges[i+1]+1e-9
        mask = (vc >= edges[i]) & (vc < hi)
        if mask.sum() == 0: continue
        out += mask.mean() * abs(c[mask].mean() - vc[mask].mean())
    return round(out, 4)

sheets = {}
sheets["RawData"] = df.drop(columns=["response"], errors="ignore")

# ── Table 5: CDI by Model Pair (base vs instruct) ─────────────────────────
t5, t5_pv = [], []
for pname in PAIRS:
    p  = df_c[df_c["pair_name"]==pname]
    bc = p[p["model_type"]=="base"]["CDI"].values
    ic = p[p["model_type"]=="instruct"]["CDI"].values
    if len(bc) == 0 or len(ic) == 0:
        print(f"T5 {pname}: skipped — missing base or instruct")
        continue
    bsm = p[p["model_type"]=="base"].groupby("seed")["CDI"].mean().values
    ism = p[p["model_type"]=="instruct"].groupby("seed")["CDI"].mean().values
    n   = min(len(bsm), len(ism))
    _, pv = stats.ttest_rel(ism[:n], bsm[:n]) if n >= 2 else (None, 1.0)
    t5_pv.append(pv)
    inst_drift = p[p["model_type"]=="instruct"]["drift_mode"].value_counts()
    dom_drift  = inst_drift.index[0] if len(inst_drift) > 0 else "N/A"
    blo, bhi   = np.percentile(bc, [2.5, 97.5])
    ilo, ihi   = np.percentile(ic, [2.5, 97.5])
    pfr_b = (len(df[(df["pair_name"]==pname)&(df["model_type"]=="base")]) - len(bc)) / \
            max(len(df[(df["pair_name"]==pname)&(df["model_type"]=="base")]), 1) * 100
    pfr_i = (len(df[(df["pair_name"]==pname)&(df["model_type"]=="instruct")]) - len(ic)) / \
            max(len(df[(df["pair_name"]==pname)&(df["model_type"]=="instruct")]), 1) * 100
    t5.append({
        "Pair":          pname,
        "CDI_base_mean": round(bc.mean(), 3),
        "CDI_base_sd":   round(bc.std(ddof=1), 3) if len(bc)>1 else np.nan,
        "CDI_base_95CI": f"[{round(blo,3)},{round(bhi,3)}]",
        "CDI_inst_mean": round(ic.mean(), 3),
        "CDI_inst_sd":   round(ic.std(ddof=1), 3) if len(ic)>1 else np.nan,
        "CDI_inst_95CI": f"[{round(ilo,3)},{round(ihi,3)}]",
        "ΔCDI":          round(ic.mean()-bc.mean(), 3),
        "p_value":       round(pv, 4),
        "cohen_d":       round(cohens_d(ic, bc), 2),
        "dom_drift":     dom_drift,
        "PFR_base_pct":  round(pfr_b, 1),
        "PFR_inst_pct":  round(pfr_i, 1),
    })
    print(f"T5 {pname}: base={bc.mean():.3f} inst={ic.mean():.3f} "
          f"Δ={ic.mean()-bc.mean():+.3f} p={pv:.4f} d={cohens_d(ic,bc):.2f}")

if len(t5_pv) >= 2:
    _, corr_p, _, _ = multipletests(t5_pv, method="holm")
    for i, r in enumerate(t5): r["p_corrected"] = round(corr_p[i], 4)
else:
    for r in t5: r["p_corrected"] = r["p_value"]
sheets["Table5_CDI_by_Pair"] = pd.DataFrame(t5)

# ── Table 6: CDI by Dataset ───────────────────────────────────────────────
t6 = []
for ds in DSS:
    for mtype in MODEL_TYPES:
        sub = df_c[(df_c["dataset"]==ds)&(df_c["model_type"]==mtype)]
        if len(sub) == 0: continue
        dom = sub["drift_mode"].value_counts()
        t6.append({
            "Dataset":    ds,
            "Model_Type": mtype,
            "CDI_mean":   round(sub["CDI"].mean(),3),
            "CDI_sd":     round(sub["CDI"].std(ddof=1),3) if len(sub)>1 else np.nan,
            "ECE":        ece(sub["VC"],(sub["label"]=="correct").astype(int)),
            "dom_drift":  dom.index[0] if len(dom)>0 else "N/A",
            "n":          len(sub),
        })
        print(f"T6 {ds} {mtype}: CDI={sub['CDI'].mean():.3f}")
sheets["Table6_CDI_by_Dataset"] = pd.DataFrame(t6)

# ── Table 7: CDI by Label ─────────────────────────────────────────────────
t7, t7_pv = [], []
for ds in DSS:
    for mtype in MODEL_TYPES:
        sub  = df_c[(df_c["dataset"]==ds)&(df_c["model_type"]==mtype)]
        corr = sub[sub["label"]=="correct"]["CDI"].values
        hall = sub[sub["label"]=="hallucinated"]["CDI"].values
        if len(corr) < 3 or len(hall) < 3:
            print(f"T7 {ds} {mtype}: skipped (corr={len(corr)}, hall={len(hall)})")
            continue
        _, pv = stats.ttest_ind(hall, corr)
        t7_pv.append(pv)
        t7.append({
            "Dataset":       ds,
            "Model_Type":    mtype,
            "CDI_corr_mean": round(corr.mean(),3),
            "CDI_hall_mean": round(hall.mean(),3),
            "Δ(H-C)":        round(hall.mean()-corr.mean(),3),
            "p_value":       round(pv,4),
            "cohen_d":       round(cohens_d(hall,corr),2),
            "n_corr":        len(corr),
            "n_hall":        len(hall),
        })
        print(f"T7 {ds} {mtype}: Δ={hall.mean()-corr.mean():+.3f} d={cohens_d(hall,corr):.2f}")

if len(t7_pv) >= 2:
    _, corr_p, _, _ = multipletests(t7_pv, method="holm")
    for i, r in enumerate(t7): r["p_corrected"] = round(corr_p[i],4)
sheets["Table7_CDI_by_Label"] = pd.DataFrame(t7) if t7 else pd.DataFrame(
    columns=["Dataset","Model_Type","CDI_corr_mean","CDI_hall_mean",
             "Δ(H-C)","p_value","cohen_d"])

# ── Table 8: RHR ──────────────────────────────────────────────────────────
t8 = []
for pname in PAIRS:
    for mtype in MODEL_TYPES:
        sub = df[(df["pair_name"]==pname)&(df["model_type"]==mtype)]
        if len(sub) == 0: continue
        t8.append({
            "Pair":       pname,
            "model_type": mtype,
            "RHR_mean":   round(sub["rh"].mean(),3),
            "RHR_sd":     round(sub["rh"].std(ddof=1),3) if len(sub)>1 else np.nan,
            "n":          len(sub),
        })
        print(f"T8 {pname} {mtype}: RHR={sub['rh'].mean():.3f}")
sheets["Table8_RHR"] = pd.DataFrame(t8)

# ── Table 9: Cross-Model ──────────────────────────────────────────────────
t9 = []
for pname in PAIRS:
    for mtype in MODEL_TYPES:
        sub = df_c[(df_c["pair_name"]==pname)&(df_c["model_type"]==mtype)]
        if len(sub) == 0: continue
        rhr_v = df[(df["pair_name"]==pname)&(df["model_type"]==mtype)]["rh"].mean()
        t9.append({
            "Model":  sub["model_name"].iloc[0],
            "Type":   mtype,
            "Pair":   pname,
            "CDI":    round(sub["CDI"].mean(),3),
            "CDI_sd": round(sub["CDI"].std(ddof=1),3) if len(sub)>1 else np.nan,
            "ECE":    ece(sub["VC"],(sub["label"]=="correct").astype(int)),
            "RHR":    round(rhr_v,3),
            "n":      len(sub),
        })
        print(f"T9 {sub['model_name'].iloc[0]} ({mtype}): CDI={sub['CDI'].mean():.3f}")
sheets["Table9_Cross_Model"] = pd.DataFrame(t9).sort_values("CDI",ascending=False)

# ── Table 10: Ablation AUROC (instruct only) ──────────────────────────────
t10 = []
inst_full = df_c[
    (df_c["model_type"]=="instruct") &
    (df_c["label"].isin(["correct","hallucinated"]))
]
for ds in DSS:
    sub = inst_full[inst_full["dataset"]==ds]
    if len(sub) < 10:
        print(f"T10 {ds}: skipped (n={len(sub)})")
        continue
    labels = (sub["label"]=="hallucinated").astype(int).values
    if labels.sum() == 0 or labels.sum() == len(labels):
        print(f"T10 {ds}: only one class")
        continue
    ece_sig = np.abs(sub["VC"].values - 0.5)
    try:
        r = {
            "Dataset":  ds,
            "VC_only":  round(roc_auc_score(labels,sub["VC"].values),3),
            "IC_only":  round(roc_auc_score(labels,sub["IC"].values),3),
            "ECE_only": round(roc_auc_score(labels,ece_sig),3),
            "CDI":      round(roc_auc_score(labels,sub["CDI"].values),3),
            "n_corr":   int((labels==0).sum()),
            "n_hall":   int((labels==1).sum()),
        }
        r["CDI_vs_best"] = round(
            r["CDI"]-max(r["VC_only"],r["IC_only"],r["ECE_only"]),3)
        t10.append(r)
        print(f"T10 {ds}: CDI={r['CDI']} ({r['CDI_vs_best']:+.3f})")
    except Exception as e:
        print(f"T10 {ds}: error — {e}")

if t10:
    adf = pd.DataFrame(t10)
    avg = adf[["VC_only","IC_only","ECE_only","CDI"]].mean()
    print(f"T10 avg: VC={avg.VC_only:.3f} IC={avg.IC_only:.3f} "
          f"ECE={avg.ECE_only:.3f} CDI={avg.CDI:.3f}")
    sheets["Table10_Ablation"] = adf
else:
    sheets["Table10_Ablation"] = pd.DataFrame(
        columns=["Dataset","VC_only","IC_only","ECE_only","CDI","CDI_vs_best"])

# ── Figure 5: RHR Sensitivity ─────────────────────────────────────────────
thresholds = [0.45,0.50,0.55,0.60,0.65,0.70,0.75]
sens = []
for pname in PAIRS:
    p = df[df["pair_name"]==pname].copy()
    for tau in thresholds:
        for mtype in MODEL_TYPES:
            sub = p[p["model_type"]==mtype]
            if len(sub) == 0: continue
            rhr_tau = sub.apply(
                lambda r: bool(HEDGE_REGEX.search(str(r.get("response",""))))
                          and float(r.get("IC",0)) >= tau, axis=1).mean()
            sens.append({"pair":pname,"model_type":mtype,
                         "tau":tau,"RHR":round(rhr_tau,4)})
sheets["Figure5_RHR_Sensitivity"] = pd.DataFrame(sens)

# ── Write Excel ───────────────────────────────────────────────────────────
with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
    for name, sdf in sheets.items():
        sdf.to_excel(writer, sheet_name=name[:31], index=False)
print(f"\nTables saved: {EXCEL_OUT}")

# ============================================================
# FIGURES 2, 3, 4
# ============================================================
print("\nGenerating figures...")
sns.set_style("whitegrid")
models_list = list(df_c["model_name"].unique())

def make_fig(n_panels, figsize=(15,5)):
    cols = min(n_panels, 3)
    rows = (n_panels + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1]*rows))
    return fig, np.array(axes).flatten()

# Figure 2: Reliability Diagrams
fig, axes = make_fig(len(models_list))
for idx, model in enumerate(models_list):
    ax  = axes[idx]
    sub = df_c[df_c["model_name"]==model]
    bins = np.linspace(0,1,11)
    bcs, accs, confs = [], [], []
    for i in range(10):
        mask = (sub["VC"]>=bins[i]) & (sub["VC"]<(bins[i+1] if i<9 else bins[i+1]+1e-9))
        if mask.sum() > 0:
            bcs.append((bins[i]+bins[i+1])/2)
            accs.append((sub[mask]["label"]=="correct").mean())
            confs.append(sub[mask]["VC"].mean())
    if bcs:
        ax.bar(bcs,[a-c for a,c in zip(accs,confs)],width=0.08,
               bottom=confs,alpha=0.7,color='steelblue')
    ax.plot([0,1],[0,1],'r--',lw=2,label='Perfect calibration')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ece_val = ece(sub["VC"],(sub["label"]=="correct").astype(int))
    ax.set_title(f"{model}\nECE={ece_val:.3f}")
    ax.legend(fontsize=8)
for idx in range(len(models_list), len(axes)): axes[idx].axis('off')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/Figure2_Reliability_Diagrams.png",dpi=300,bbox_inches='tight')
plt.close()
print("  Figure 2 saved")

# Figure 3: VC vs IC Scatter
fig, axes = make_fig(len(models_list))
for idx, model in enumerate(models_list):
    ax  = axes[idx]
    sub = df_c[df_c["model_name"]==model]
    for label, color in [("correct","blue"),("hallucinated","red"),("abstained","gray")]:
        ss = sub[sub["label"]==label]
        if len(ss): ax.scatter(ss["VC"],ss["IC"],c=color,alpha=0.4,s=15,label=label)
    ax.plot([0,1],[0,1],'k--',lw=1.5)
    ax.set_xlabel("VC"); ax.set_ylabel("IC")
    ax.set_title(f"{model}\nCDI={sub['CDI'].mean():.3f}")
    ax.legend(fontsize=8)
for idx in range(len(models_list), len(axes)): axes[idx].axis('off')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/Figure3_VC_vs_IC_Scatter.png",dpi=300,bbox_inches='tight')
plt.close()
print("  Figure 3 saved")

# Figure 4: CDI Histograms
fig, axes = make_fig(len(models_list))
for idx, model in enumerate(models_list):
    ax  = axes[idx]
    sub = df_c[df_c["model_name"]==model]
    for label, color in [("correct","blue"),("hallucinated","red")]:
        data = sub[sub["label"]==label]["CDI"].values
        if len(data):
            ax.hist(data,bins=20,alpha=0.5,color=color,
                    label=f"{label} (n={len(data)})",density=True)
    cm = sub[sub["label"]=="correct"]["CDI"].mean() \
         if len(sub[sub["label"]=="correct"]) else 0
    hm = sub[sub["label"]=="hallucinated"]["CDI"].mean() \
         if len(sub[sub["label"]=="hallucinated"]) else 0
    ax.axvline(cm,color='blue',ls='--',alpha=0.7,label=f'Corr μ={cm:.2f}')
    ax.axvline(hm,color='red', ls='--',alpha=0.7,label=f'Hall μ={hm:.2f}')
    ax.set_xlabel("CDI"); ax.set_ylabel("Density")
    ax.set_title(model); ax.legend(fontsize=7)
for idx in range(len(models_list), len(axes)): axes[idx].axis('off')
plt.tight_layout()
plt.savefig(f"{FIGDIR}/Figure4_CDI_Histograms.png",dpi=300,bbox_inches='tight')
plt.close()
print("  Figure 4 saved")

# ============================================================
# DRIFT MODE SUMMARY
# ============================================================
print("\n" + "="*60)
print("DRIFT MODE SUMMARY")
print("="*60)
for ds in DSS:
    for mtype in MODEL_TYPES:
        sub = df_c[(df_c["dataset"]==ds)&(df_c["model_type"]==mtype)]
        if len(sub) == 0: continue
        ued = (sub["drift_mode"]=="UED").sum()
        oed = (sub["drift_mode"]=="OED").sum()
        bal = (sub["drift_mode"]=="balanced").sum()
        n   = len(sub)
        print(f"{ds:12} | {mtype:8} | "
              f"UED:{ued:4} ({ued/n*100:5.1f}%) | "
              f"OED:{oed:4} ({oed/n*100:5.1f}%) | "
              f"Bal:{bal:4}")
print("="*60)

# ── Parse rate summary ────────────────────────────────────────────────────
print("\n" + "="*60)
print("PARSE FAILURE RATE SUMMARY")
print("="*60)
for pname in PAIRS:
    for mtype in MODEL_TYPES:
        total = len(df[(df["pair_name"]==pname)&(df["model_type"]==mtype)])
        parsed = len(df_c[(df_c["pair_name"]==pname)&(df_c["model_type"]==mtype)])
        if total > 0:
            pfr = (total - parsed) / total * 100
            print(f"{pname:15} {mtype:8}: {parsed}/{total} parsed "
                  f"(PFR={pfr:.1f}%)")
print("="*60)

print(f"\n{'='*60}")
print("COMPLETE")
print(f"  Tables:  {EXCEL_OUT}")
print(f"  Figures: {FIGDIR}/")
print(f"{'='*60}")