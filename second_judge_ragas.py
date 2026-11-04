# -*- coding: utf-8 -*-
"""
second_judge_ragas.py
=====================

Second-judge RAGAS re-scoring for Reviewer 1, Concern 4:

    "The evaluation relies on a single LLM-based judge (GPT-4.1-mini).
     Comparing its results with another independent evaluation model or
     human assessment would improve confidence in the reported RAGAS scores."

A stratified random sub-sample of the archived 1,000-question evaluation set
is re-scored with a second, independent judge LLM. Everything except the judge
model follows the code path of 04_RAGAS_statistical-analysis.ipynb: RAGAS 0.4.1
metric objects from ragas.metrics (strictness=1), ChatOpenAI via OpenRouter
(temperature=0.0, max_tokens=1024, JSON response format, same headers), local
intfloat/multilingual-e5-large embeddings, and the same RunConfig. No answer is
regenerated: questions, answers, retrieved contexts and references are read
from the archived per-question RAGAS file.

Which metrics are re-judged
---------------------------
answer_relevancy, faithfulness, answer_correctness
    depend on both the judge and the generated answer: re-judged for all three
    models on the same questions, so model comparisons stay paired.
answer_similarity
    embeddings only in RAGAS 0.4.1 (SemanticSimilarity: cosine of the e5
    embeddings of reference and answer, no LLM call). It is recomputed at zero
    API cost only to confirm that the embedding path is identical.
context_recall, context_precision
    inputs (question, retrieved contexts, reference) are identical for the
    three models, so these metrics cannot separate models. With
    --context-metrics once (default) each sampled question is judged once;
    with --context-metrics skip they are omitted.

Cost control
------------
estimate  renders the exact RAGAS prompts for the sample, counts tokens and
          prices them with live OpenRouter prices. No API key, no spend.
pilot     the first PILOT_QUESTIONS questions of the sample (x 3 models).
          Reports measured tokens and cost per question and extrapolates.
run       the full sub-sample. Questions already scored are skipped.

* Every judge response is cached on disk, keyed by a hash of the exact request,
  and every finished (question, model, metric) score is cached, so a crash or a
  rerun never pays for the same request twice. The pilot questions are the
  first questions of the run sample, so the pilot spend is reused by the run.
  Identical requests are also sent only once (for example the reference
  statements of answer_correctness, which are the same for the three models,
  or two models giving the same abstention answer); the reused response is the
  judgment the judge gave to that exact input.
* --budget-usd is a cumulative cap for the output directory (pilot included)
  and can never exceed HARD_BUDGET_CAP_USD. Before every request the
  worst-case cost of that request is reserved; a request that could cross the
  cap is not sent, the run stops, and the unfinished question is not saved.
* Transient API errors (429, 5xx, timeouts, connection errors) are retried with
  exponential backoff. A question that still has a transport failure is not
  saved and is retried on the next run. Judge failures (unparsable output,
  length limit) are kept as NaN, as in the original run.

Usage (from the repository root)
--------------------------------
python second_judge_ragas.py estimate
python second_judge_ragas.py pilot --budget-usd 0.10
python second_judge_ragas.py run --n-questions 150 --budget-usd 1.00
python second_judge_compare.py

Outputs (in --out-dir, default arabicaqa_rag_results/second_judge/<judge>)
--------------------------------------------------------------------------
sample_questions.csv            stratified, nested question order
cache/llm_responses.jsonl       raw judge responses (request-hash keyed)
cache/scores.jsonl              finished scores
cost_ledger.jsonl               one line per billed request
second_judge_per_question.csv   question x model scores
second_judge_context_once.csv   question-level context_recall / precision
run_summary.json                tokens, cost, NaN counts, extrapolation
"""

import argparse
import ast
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

# RAGAS usage telemetry makes network calls on every LLM call; it does not
# affect scores. Must be set before ragas is imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# ============================================================
# CONFIG
# ============================================================
PER_QUESTION_CSV = os.path.join(
    "arabicaqa_rag_results", "ragas_full_llama_mistral_command",
    "ragas_per_question_all_models.csv",
)
OUT_ROOT = os.path.join("arabicaqa_rag_results", "second_judge")

ORIGINAL_JUDGE = "openai/gpt-4.1-mini"
JUDGE_MODEL = "qwen/qwen3-235b-a22b-2507"      # Qwen3-235B-A22B-Instruct-2507
JUDGE_PROVIDER = "deepinfra"                   # OpenRouter provider slug; "" = let OpenRouter route
JUDGE_QUANTIZATION = "fp8"                     # "" = any
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EXPECTED_RAGAS_VERSION = "0.4.1"

# Absolute ceiling on cumulative spend per output directory. --budget-usd can
# lower it but never raise it. Edit only deliberately.
HARD_BUDGET_CAP_USD = 5.00

SAMPLE_SEED = 42
DEFAULT_N_QUESTIONS = 150
PILOT_QUESTIONS = 6
MAX_TOKENS = 1024               # as in the original judge configuration
EXTRAPOLATE_N = [100, 150, 200, 300, 1000]

MODEL_KEYS = ["command", "llama", "mistral"]
REJUDGED_METRICS = ["answer_relevancy", "faithfulness", "answer_correctness"]
EMBEDDING_ONLY_METRICS = ["answer_similarity"]
CONTEXT_METRICS = ["context_recall", "context_precision"]
QUESTION_LEVEL_KEY = "_question"    # model_key used for question-level scores

# Transport retries (in addition to the OpenAI client's own retries).
TRANSPORT_ATTEMPTS = 6
BACKOFF_BASE_SEC = 4.0
BACKOFF_MAX_SEC = 90.0
MAX_CONSECUTIVE_FAILED_QUESTIONS = 3

# Instruction prefixes of the RAGAS 0.4.1 prompts, used only to label the
# measured tokens per call type in the pilot report.
PROMPT_KINDS = [
    ("Generate a question for the given answer", "answer_relevancy.question_gen"),
    ("Given a question and an answer, analyze the complexity", "statement_generation"),
    ("Your task is to judge the faithfulness", "faithfulness.nli"),
    ("Given a ground truth and an answer statements", "answer_correctness.classify"),
    ("Given a context, and an answer, analyze each sentence", "context_recall.classify"),
    ("Given question, answer and context verify", "context_precision.verify"),
    ("The output string did not satisfy", "fix_output_format"),
]


# ============================================================
# SMALL HELPERS
# ============================================================
def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def judge_slug(model_id):
    return model_id.replace("/", "__").replace(":", "_")


def prompt_kind(text):
    head = text[:400]
    for prefix, name in PROMPT_KINDS:
        if prefix.lower() in head.lower():
            return name
    return "other"


def parse_context_list(x):
    """Same parsing as 04_RAGAS_statistical-analysis.ipynb."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(c).strip() for c in x if str(c).strip()]
    s = str(x).strip()
    if not s or s in ["[]", "nan", "None"]:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except Exception:
            return []
    return [s]


def append_jsonl(path, record, lock):
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash can leave one truncated last line; ignore it.
                continue
    return rows


# ============================================================
# DATA AND SAMPLE
# ============================================================
def load_archive(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run from the repository root and make sure the "
            "Git LFS file is checked out (git lfs pull)."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) < 100 or "question_id" not in df.columns:
        raise ValueError(f"{path} does not look like the per-question RAGAS file "
                         "(is it still a Git LFS pointer?).")
    df["is_answerable"] = df["is_answerable"].astype(bool)
    df["answer"] = df["answer"].fillna("").astype(str)
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str)
    df["question"] = df["question"].fillna("").astype(str)
    df["contexts_list"] = df["retrieved_contexts"].apply(parse_context_list)
    missing = set(MODEL_KEYS) - set(df["model_key"].unique())
    if missing:
        raise ValueError(f"Models missing from archive: {missing}")
    return df


def sample_order(df, seed):
    """Stratified, nested question order.

    Answerable and unanswerable question ids are each shuffled with `seed`
    and interleaved (A, U, A, U, ...). The first n questions of this order
    are therefore always n/2 answerable + n/2 unanswerable, and every smaller
    sample (the pilot included) is a prefix of every larger one.
    """
    q = df.drop_duplicates("question_id")[["question_id", "is_answerable"]]
    rng = np.random.default_rng(seed)
    ans = rng.permutation(np.sort(q.loc[q["is_answerable"], "question_id"].to_numpy()))
    una = rng.permutation(np.sort(q.loc[~q["is_answerable"], "question_id"].to_numpy()))
    order = []
    for a, u in zip(ans, una):
        order.extend([int(a), int(u)])
    out = pd.DataFrame({"rank": np.arange(1, len(order) + 1), "question_id": order})
    out = out.merge(q, on="question_id", how="left")
    return out


# ============================================================
# OPENROUTER PRICES
# ============================================================
def fetch_endpoints(model_id, timeout=30):
    url = f"{OPENROUTER_BASE_URL}/models/{model_id}/endpoints"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)["data"]
    eps = []
    for e in data.get("endpoints", []):
        p = e.get("pricing", {})
        eps.append({
            "provider_name": e.get("provider_name"),
            "tag": e.get("tag"),
            "quantization": e.get("quantization"),
            "price_in": float(p.get("prompt", 0) or 0) * 1e6,
            "price_out": float(p.get("completion", 0) or 0) * 1e6,
            "response_format": "response_format" in (e.get("supported_parameters") or []),
            "max_completion_tokens": e.get("max_completion_tokens"),
            "status": e.get("status"),
        })
    if model_id.endswith(":free"):
        eps = [dict(e, price_in=0.0, price_out=0.0) for e in eps]
    return eps


def select_prices(model_id, provider, quantization, policy="max"):
    """Return (price_in, price_out, description) in USD per million tokens.

    policy="max": the highest price among matching JSON-capable endpoints, so
    budget reservations stay conservative wherever OpenRouter routes.
    policy="cheapest_pinnable": the cheapest fp8/bf16 endpoint, i.e. the one a
    run would pin (used only to compare candidate judges in estimate mode).
    """
    eps = [e for e in fetch_endpoints(model_id) if e["response_format"]]
    if provider:
        eps = [e for e in eps if (e["tag"] or "").split("/")[0] == provider
               or (e["provider_name"] or "").lower() == provider.lower()]
    if quantization:
        eps = [e for e in eps if (e["quantization"] or "") == quantization]
    if policy == "cheapest_pinnable" and not model_id.endswith(":free"):
        good = [e for e in eps if e["quantization"] in ("fp8", "bf16", "fp16")]
        if good:
            eps = [min(good, key=lambda e: (e["price_in"] + 4 * e["price_out"]))]
    if not eps:
        raise RuntimeError(
            f"No JSON-capable endpoint for {model_id} matching provider='{provider}' "
            f"quantization='{quantization}'. Pass --provider '' or --price-in/--price-out."
        )
    p_in = max(e["price_in"] for e in eps)
    p_out = max(e["price_out"] for e in eps)
    desc = ", ".join(f"{e['provider_name']} {e['quantization']} "
                     f"${e['price_in']:.3f}/${e['price_out']:.3f}" for e in eps)
    return p_in, p_out, desc


def provider_routing(provider, quantization):
    routing = {"require_parameters": True}
    if provider:
        routing.update({"order": [provider], "allow_fallbacks": False})
    if quantization:
        routing["quantizations"] = [quantization]
    return routing


def openrouter_key_info(api_key):
    """Credit information of the API key, or None. Never prints the key."""
    try:
        req = urllib.request.Request(f"{OPENROUTER_BASE_URL}/key",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r).get("data", {})
    except Exception:
        return None


# ============================================================
# BUDGET LEDGER AND RESPONSE CACHE
# ============================================================
class BudgetExceeded(Exception):
    pass


class FatalAPIError(Exception):
    pass


class TransportFailure(Exception):
    pass


class JudgeLengthLimit(Exception):
    pass


class Ledger:
    """Cumulative spend for one output directory, with worst-case reservation."""

    def __init__(self, path, cap_usd, price_in, price_out):
        self.path = path
        self.cap = cap_usd
        self.price_in = price_in
        self.price_out = price_out
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()
        prior = read_jsonl(path)
        self.spent_before = float(sum(r.get("cost_usd", 0.0) for r in prior))
        self.spent = self.spent_before
        self.reserved = 0.0
        self.stopped = False
        self.stop_reason = ""
        self.fatal = ""
        self.run_records = []

    def worst_case(self, prompt_text):
        # Byte-level BPE tokenizers never use more tokens than UTF-8 bytes;
        # 64 tokens cover the chat template. Completion is capped by max_tokens.
        tokens_in = len(prompt_text.encode("utf-8")) + 64
        return (tokens_in * self.price_in + MAX_TOKENS * self.price_out) / 1e6

    def reserve(self, amount):
        with self.lock:
            if self.stopped:
                raise BudgetExceeded(self.stop_reason or "run stopped")
            if self.spent + self.reserved + amount > self.cap:
                self.stopped = True
                self.stop_reason = (
                    f"budget cap ${self.cap:.4f} reached: spent ${self.spent:.4f}, "
                    f"in flight ${self.reserved:.4f}, next request up to ${amount:.4f}"
                )
                raise BudgetExceeded(self.stop_reason)
            self.reserved += amount

    def settle(self, reserved, record):
        with self.lock:
            self.reserved -= reserved
            self.spent += record["cost_usd"]
            self.run_records.append(record)
        append_jsonl(self.path, record, self.file_lock)

    def release(self, reserved):
        with self.lock:
            self.reserved -= reserved

    def stop(self, reason, fatal=False):
        with self.lock:
            self.stopped = True
            if fatal:
                self.fatal = reason
            if not self.stop_reason:
                self.stop_reason = reason


class ResponseCache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.data = {r["key"]: r for r in read_jsonl(path) if "key" in r}
        self.inflight = set()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        return self.data.get(key)

    def put(self, record):
        # Written to disk before the response is used, so a crash right after
        # a billed request never loses it.
        append_jsonl(self.path, record, self.file_lock)
        with self.lock:
            self.data[record["key"]] = record


class ScoreCache:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        for r in read_jsonl(path):
            self.data[(int(r["question_id"]), r["model_key"], r["metric"])] = r["value"]

    def has(self, qid, model_key, metric):
        return (int(qid), model_key, metric) in self.data

    def put_many(self, records):
        with self.lock:
            for r in records:
                self.data[(int(r["question_id"]), r["model_key"], r["metric"])] = r["value"]
            with open(self.path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())


# Per-batch state, reset before every ragas.evaluate call.
BATCH = {"transport_error": False, "errors": []}


# ============================================================
# JUDGE LLM: the original ChatOpenAI configuration + cache + budget
# ============================================================
def build_judge(args, api_key, ledger, cache, rate_limiter):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI
    import openai

    routing = provider_routing(args.provider, args.quantization)
    extra_body = {"provider": routing}
    if args.reasoning_off:
        # For judges with an optional thinking mode (e.g. DeepSeek V4, Gemma 4):
        # thinking tokens are billed and count against max_tokens.
        extra_body["reasoning"] = {"enabled": False}
    routing_key = json.dumps(extra_body, sort_keys=True)

    class CachedBudgetedChatOpenAI(ChatOpenAI):
        """ChatOpenAI with a disk cache and a hard budget around each request.

        Only _agenerate/_generate are wrapped; the request itself is built and
        sent by ChatOpenAI exactly as in the original notebook.
        """

        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            body = response if isinstance(response, dict) else response.model_dump()
            if result.llm_output is not None:
                result.llm_output["provider"] = body.get("provider")
            return result

        def _request_key(self, messages, stop):
            payload = [
                self.model_name, routing_key, self.temperature, self.max_tokens,
                self.n, self.model_kwargs, stop,
                [(m.type, m.content) for m in messages],
            ]
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        @staticmethod
        def _from_cache(rec):
            if rec["status"] == "length":
                raise JudgeLengthLimit("cached: judge output hit max_tokens")
            msg = AIMessage(content=rec["text"])
            gen = ChatGeneration(message=msg,
                                 generation_info={"finish_reason": rec.get("finish_reason") or "stop"})
            return ChatResult(generations=[gen],
                              llm_output={"token_usage": None, "model_name": rec.get("model"),
                                          "cache_hit": True})

        def _record(self, key, kind, usage, status, model, provider, finish_reason, t0):
            usage = usage or {}
            p_tok = int(usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            if usage.get("cost") is not None:
                cost, source = float(usage["cost"]), "openrouter"
            else:
                cost = (p_tok * ledger.price_in + c_tok * ledger.price_out) / 1e6
                source = "computed"
            return {
                "ts": utc_now(), "key": key, "phase": args.phase_label, "kind": kind,
                "model": model, "provider": provider, "status": status,
                "finish_reason": finish_reason, "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "reasoning_tokens": int((details or {}).get("reasoning_tokens") or 0),
                "cost_usd": cost, "cost_source": source,
                "latency_sec": round(time.time() - t0, 3),
            }

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                return await self._agenerate_guarded(messages, stop, run_manager, **kwargs)
            except asyncio.CancelledError:
                # ragas timed the job out: not a judge failure, redo next run.
                BATCH["transport_error"] = True
                BATCH["errors"].append("cancelled (ragas job timeout)")
                raise

        async def _agenerate_guarded(self, messages, stop=None, run_manager=None, **kwargs):
            key = self._request_key(messages, stop)
            prompt_text = "\n".join(str(m.content) for m in messages)
            kind = prompt_kind(prompt_text)

            # Identical requests in flight (e.g. the reference statements of
            # answer_correctness, shared by the three models) wait for the first.
            while key in cache.inflight and cache.get(key) is None and not ledger.stopped:
                await asyncio.sleep(0.25)
            rec = cache.get(key)
            if rec is not None:
                cache.hits += 1
                return self._from_cache(rec)

            if ledger.fatal:
                raise FatalAPIError(ledger.fatal)
            reserved = 0.0
            settled = False
            cache.inflight.add(key)
            try:
                amount = ledger.worst_case(prompt_text)
                ledger.reserve(amount)      # raises BudgetExceeded before anything is sent
                reserved = amount
                last_err = None
                for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
                    await rate_limiter.wait()
                    t0 = time.time()
                    try:
                        result = await super()._agenerate(messages, stop=stop,
                                                          run_manager=run_manager, **kwargs)
                    except openai.LengthFinishReasonError as e:
                        completion = getattr(e, "completion", None)
                        usage = completion.model_dump().get("usage") if completion is not None else None
                        model = getattr(completion, "model", None)
                        record = self._record(key, kind, usage, "length", model,
                                              getattr(completion, "provider", None), "length", t0)
                        if record["cost_usd"] == 0.0 and not usage:
                            record["cost_usd"], record["cost_source"] = reserved, "reserved_upper_bound"
                        ledger.settle(reserved, record)
                        settled = True
                        cache.put({"key": key, "status": "length", "text": "",
                                   "finish_reason": "length", "model": model, "kind": kind})
                        cache.misses += 1
                        raise JudgeLengthLimit("judge output hit max_tokens") from e
                    except (openai.AuthenticationError, openai.PermissionDeniedError,
                            openai.NotFoundError, openai.BadRequestError) as e:
                        ledger.stop(f"fatal API error: {type(e).__name__}: {e}", fatal=True)
                        raise FatalAPIError(str(e)) from e
                    except openai.APIStatusError as e:
                        if e.status_code == 402:
                            ledger.stop("OpenRouter returned 402 (credits or key limit exhausted)",
                                        fatal=True)
                            raise FatalAPIError(str(e)) from e
                        last_err = e
                        retry_after = None
                        try:
                            retry_after = float(e.response.headers.get("retry-after"))
                        except Exception:
                            pass
                    except (openai.APIConnectionError, openai.APITimeoutError) as e:
                        last_err, retry_after = e, None
                    else:
                        gen = result.generations[0]
                        out = result.llm_output or {}
                        record = self._record(key, kind, out.get("token_usage"), "ok",
                                              out.get("model_name"), out.get("provider"),
                                              (gen.generation_info or {}).get("finish_reason"), t0)
                        ledger.settle(reserved, record)
                        settled = True
                        cache.put({"key": key, "status": "ok", "text": gen.message.content,
                                   "finish_reason": (gen.generation_info or {}).get("finish_reason"),
                                   "model": out.get("model_name"), "provider": out.get("provider"),
                                   "kind": kind})
                        cache.misses += 1
                        return result
                    if attempt < TRANSPORT_ATTEMPTS:
                        wait = min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * 2 ** (attempt - 1))
                        if retry_after:
                            wait = max(wait, retry_after)
                        await asyncio.sleep(wait * (0.75 + 0.5 * random.random()))
                BATCH["transport_error"] = True
                BATCH["errors"].append(f"{type(last_err).__name__}: {last_err}")
                raise TransportFailure(str(last_err))
            except asyncio.CancelledError:
                # A request may have been billed without a response: book its
                # worst-case cost so the cap stays conservative.
                if reserved and not settled:
                    ledger.settle(reserved, {
                        "ts": utc_now(), "key": key, "phase": args.phase_label, "kind": kind,
                        "status": "cancelled", "prompt_tokens": 0, "completion_tokens": 0,
                        "cost_usd": reserved, "cost_source": "reserved_upper_bound"})
                    settled = True
                raise
            finally:
                if reserved and not settled:
                    ledger.release(reserved)
                cache.inflight.discard(key)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # RAGAS uses the async path; keep the sync path cached and budgeted too.
            return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    return CachedBudgetedChatOpenAI(
        model=args.judge,
        temperature=0.0,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_tokens=MAX_TOKENS,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body=extra_body,
        default_headers={
            "HTTP-Referer": "https://your-project-name.com",
            "X-Title": "ArabicaQA RAGAS",
        },
    )


class RateLimiter:
    """Minimum spacing between request starts (needed for :free models)."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.next_t = 0.0
        self.lock = threading.Lock()

    async def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_t)
            self.next_t = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


# ============================================================
# RAGAS METRICS (as in the notebook)
# ============================================================
def ragas_metrics(names):
    import ragas.metrics as rm
    metrics = []
    for name in names:
        m = getattr(rm, name, None)
        if m is None:
            raise RuntimeError(f"ragas.metrics.{name} not found in this ragas version")
        if hasattr(m, "strictness"):
            m.strictness = 1
        metrics.append(m)
    return metrics


def check_versions(allow_mismatch):
    import ragas
    if ragas.__version__ != EXPECTED_RAGAS_VERSION:
        msg = (f"ragas {ragas.__version__} installed; the original run used "
               f"{EXPECTED_RAGAS_VERSION} (requirements.txt).")
        if not allow_mismatch:
            raise SystemExit(msg + " Install it or pass --allow-version-mismatch.")
        print("WARNING:", msg)
    return ragas.__version__


# ============================================================
# ESTIMATE MODE (no API calls)
# ============================================================
def estimate(args, df, order):
    import tiktoken
    from ragas.metrics._answer_correctness import QuestionAnswerGroundTruth
    from ragas.metrics._answer_relevance import ResponseRelevanceInput
    from ragas.metrics._context_precision import QAC
    from ragas.metrics._context_recall import QCA
    from ragas.metrics._faithfulness import NLIStatementInput, StatementGeneratorInput

    enc = tiktoken.get_encoding("o200k_base")    # GPT-4.1-mini tokenizer

    def tok(s):
        return len(enc.encode(s))

    ar, faith, ac, cr, cp = ragas_metrics(REJUDGED_METRICS + CONTEXT_METRICS)
    qids = order["question_id"].head(args.estimate_questions).tolist()
    sub = df[df["question_id"].isin(qids)]

    per_q = []
    for qid, g in sub.groupby("question_id"):
        r0 = g.iloc[0]
        q, ref, ctx = r0["question"], r0["ground_truth"], r0["contexts_list"]
        tin = tout = 0
        calls = 0
        # Reference statements of answer_correctness: identical for the three
        # models, so sent once per question (shared through the cache).
        ref_prompt = ac.statement_generator_prompt.to_string(
            StatementGeneratorInput(question=q, answer=ref))
        tin += tok(ref_prompt) + 8
        tout += tok(q) + tok(ref) + 20
        calls += 1
        for _, r in g.iterrows():
            a = r["answer"]
            prompts = [
                ar.question_generation.to_string(ResponseRelevanceInput(response=a)),
                faith.statement_generator_prompt.to_string(
                    StatementGeneratorInput(question=q, answer=a)),
                faith.nli_statements_prompt.to_string(
                    NLIStatementInput(context="\n".join(ctx), statements=[a])),
                ac.statement_generator_prompt.to_string(
                    StatementGeneratorInput(question=q, answer=a)),
                ac.correctness_prompt.to_string(
                    QuestionAnswerGroundTruth(question=q, answer=[a], ground_truth=[ref])),
            ]
            tin += sum(tok(p) + 8 for p in prompts)
            # Rough output sizes: generated question, statements, verdicts with reasons.
            tout += (tok(q) + 25) + 2 * (tok(q) + tok(a) + 20) + (tok(q) + tok(a) + 70) \
                + (tok(a) + tok(ref) + 2 * tok(q) + 90)
            calls += len(prompts)
        rec = {"question_id": qid, "rejudge_in": tin, "rejudge_out": tout, "rejudge_calls": calls}
        cr_prompt = cr.context_recall_prompt.to_string(
            QCA(question=q, context="\n".join(ctx), answer=ref))
        cp_prompts = [cp.context_precision_prompt.to_string(
            QAC(question=q, context=c, answer=ref)) for c in ctx]
        rec["ctx_in"] = tok(cr_prompt) + 8 + sum(tok(p) + 8 for p in cp_prompts)
        rec["ctx_out"] = (tok(ref) + 60) + len(cp_prompts) * 60
        rec["ctx_calls"] = 1 + len(cp_prompts)
        rec["bytes_max"] = max(len(p.encode("utf-8")) for p in cp_prompts + [cr_prompt, ref_prompt])
        per_q.append(rec)
    e = pd.DataFrame(per_q)
    mean = e.mean(numeric_only=True)

    print("=" * 78)
    print(f"ESTIMATE (no API calls). Prompts rendered with ragas for "
          f"{len(e)} sampled questions x 3 models; tokens counted with o200k_base.")
    print(f"Per question: re-judged metrics {mean.rejudge_in:,.0f} in / ~{mean.rejudge_out:,.0f} "
          f"out tokens in {mean.rejudge_calls:.0f} calls; context metrics once "
          f"{mean.ctx_in:,.0f} in / ~{mean.ctx_out:,.0f} out in {mean.ctx_calls:.0f} calls.")
    print(f"Tokenizer factor for the judge: x{args.tokenizer_factor} on input; "
          "high estimate = 2 x point estimate.")
    print("=" * 78)

    judges = args.compare_judges.split(",") if args.compare_judges else [args.judge]
    rows = []
    for j in judges:
        j = j.strip()
        prov = args.provider if j == args.judge else ""
        quant = args.quantization if j == args.judge else ""
        try:
            p_in, p_out, desc = select_prices(
                j, prov, quant, policy="max" if j == args.judge else "cheapest_pinnable")
        except Exception as ex:
            print(f"{j}: price lookup failed ({ex})")
            continue
        cin = mean.rejudge_in * args.tokenizer_factor * p_in / 1e6
        cout = mean.rejudge_out * p_out / 1e6
        xin = mean.ctx_in * args.tokenizer_factor * p_in / 1e6
        xout = mean.ctx_out * p_out / 1e6
        calls_q = mean.rejudge_calls + mean.ctx_calls
        worst_call = ((mean.bytes_max + 64) * p_in + MAX_TOKENS * p_out) / 1e6
        for n in EXTRAPOLATE_N + [PILOT_QUESTIONS]:
            point = n * (cin + cout)
            ctx = n * (xin + xout)
            rows.append({
                "judge": j, "n_questions": n, "point_usd": point,
                "high_usd": 2 * point, "context_once_point_usd": ctx,
                "calls": int(round(n * calls_q)),
                "absolute_ceiling_usd": n * calls_q * worst_call,
                "prices": desc,
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for j, g in out.groupby("judge", sort=False):
        print(f"\n{j}   [{g['prices'].iloc[0]}]")
        print(g.drop(columns=["judge", "prices"]).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        if j.endswith(":free"):
            calls = g.loc[g.n_questions == args.n_questions, "calls"]
            if len(calls):
                c = int(calls.iloc[0])
                print(f"   :free limits (OpenRouter): 20 req/min; 1,000 req/day with >= 10 credits "
                      f"purchased (else 50/day). {c:,} calls -> >= {c / 1000:.1f} days "
                      f"(or {c / 50:.0f} days at 50/day).")
    path = os.path.join(args.out_dir, "cost_estimate.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {path}")
    print("'absolute_ceiling_usd' = every call at its byte-length prompt bound and the full "
          f"{MAX_TOKENS}-token completion; real spend is far lower.")


# ============================================================
# PILOT / RUN
# ============================================================
def run(args, df, order):
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY environment variable is not set.")

    ragas_version = check_versions(args.allow_version_mismatch)

    budget = float(args.budget_usd)
    if budget > HARD_BUDGET_CAP_USD:
        raise SystemExit(f"--budget-usd {budget} exceeds HARD_BUDGET_CAP_USD "
                         f"{HARD_BUDGET_CAP_USD} in the script.")
    if args.price_in is not None and args.price_out is not None:
        p_in, p_out, price_desc = args.price_in, args.price_out, "from command line"
    else:
        p_in, p_out, price_desc = select_prices(args.judge, args.provider, args.quantization)

    os.makedirs(os.path.join(args.out_dir, "cache"), exist_ok=True)
    # Cached scores are only valid for one judge configuration.
    fingerprint = {
        "judge": args.judge, "provider_routing": provider_routing(args.provider, args.quantization),
        "reasoning_off": bool(args.reasoning_off), "max_tokens": MAX_TOKENS,
        "ragas_version": ragas_version, "embedding_model": EMBEDDING_MODEL,
        "sample_seed": SAMPLE_SEED, "temperature": 0.0,
    }
    fp_path = os.path.join(args.out_dir, "run_config.json")
    if os.path.exists(fp_path):
        old = json.load(open(fp_path, encoding="utf-8"))
        if old != fingerprint:
            raise SystemExit(
                f"{args.out_dir} was created with a different judge configuration:\n"
                f"  before: {old}\n  now:    {fingerprint}\n"
                "Use a new --out-dir so that cached scores are not mixed.")
    else:
        with open(fp_path, "w", encoding="utf-8") as f:
            json.dump(fingerprint, f, indent=2)
    ledger = Ledger(os.path.join(args.out_dir, "cost_ledger.jsonl"), budget, p_in, p_out)
    cache = ResponseCache(os.path.join(args.out_dir, "cache", "llm_responses.jsonl"))
    scores = ScoreCache(os.path.join(args.out_dir, "cache", "scores.jsonl"))
    if ledger.spent_before >= budget:
        raise SystemExit(f"Cumulative spend ${ledger.spent_before:.4f} already reaches "
                         f"the budget ${budget:.4f}.")

    key_before = openrouter_key_info(api_key)
    if key_before and key_before.get("limit_remaining") is not None:
        if float(key_before["limit_remaining"]) < budget - ledger.spent_before:
            print(f"Note: the API key's own credit limit "
                  f"(${float(key_before['limit_remaining']):.2f} left) is below this budget.")

    n = PILOT_QUESTIONS if args.mode == "pilot" else args.n_questions
    qids = order["question_id"].head(n).tolist()
    args.phase_label = args.mode

    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.run_config import RunConfig

    judge = build_judge(args, api_key, ledger, cache, RateLimiter(args.rpm))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    run_config = RunConfig(timeout=300, max_retries=2, max_wait=60, max_workers=args.max_workers)

    main_names = REJUDGED_METRICS + EMBEDDING_ONLY_METRICS
    main_metrics = ragas_metrics(main_names)
    ctx_names = CONTEXT_METRICS if args.context_metrics == "once" else []
    ctx_metrics = ragas_metrics(ctx_names) if ctx_names else []

    print("=" * 78)
    print(f"SECOND-JUDGE RAGAS  mode={args.mode}  questions={n}  judge={args.judge}")
    print(f"provider routing={provider_routing(args.provider, args.quantization)}")
    print(f"prices (USD/M tokens) in={p_in:.3f} out={p_out:.3f}  [{price_desc}]")
    print(f"budget ${budget:.2f} (hard cap ${HARD_BUDGET_CAP_USD:.2f}); "
          f"already spent in this directory ${ledger.spent_before:.4f}")
    print(f"metrics: {main_names} x 3 models; context metrics: "
          f"{ctx_names or 'skipped'} once per question; ragas {ragas_version}")
    print("=" * 78)

    def run_batch(rows, metrics, names, model_keys):
        BATCH["transport_error"] = False
        BATCH["errors"] = []
        ds = Dataset.from_dict({
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts_list"] for r in rows],
            "ground_truth": [r["ground_truth"] for r in rows],
        })
        result = evaluate(ds, metrics=metrics, llm=judge, embeddings=embeddings,
                          run_config=run_config, raise_exceptions=False,
                          show_progress=False)
        if ledger.stopped or BATCH["transport_error"]:
            return None
        sc = result.to_pandas()
        recs = []
        for i, mk in enumerate(model_keys):
            for name in names:
                v = sc[name].iloc[i] if name in sc.columns else np.nan
                v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                recs.append({"question_id": int(rows[i]["question_id"]), "model_key": mk,
                             "metric": name, "value": v, "judge": args.judge,
                             "ts": utc_now()})
        return recs

    done_q = skipped_q = failed_q = consecutive_failures = 0
    failures = []
    t_start = time.time()
    try:
        for i, qid in enumerate(qids, 1):
            if ledger.stopped:
                break
            g = df[df["question_id"] == qid].set_index("model_key").loc[MODEL_KEYS]
            rows = [dict(r, question_id=qid) for _, r in g.reset_index().iterrows()]
            need_main = any(not scores.has(qid, mk, m) for mk in MODEL_KEYS for m in main_names)
            need_ctx = any(not scores.has(qid, QUESTION_LEVEL_KEY, m) for m in ctx_names)
            if not need_main and not need_ctx:
                skipped_q += 1
                continue
            ok = True
            if need_main:
                recs = run_batch(rows, main_metrics, main_names, MODEL_KEYS)
                if recs is None:
                    ok = False
                else:
                    scores.put_many(recs)
            if ok and need_ctx and not ledger.stopped:
                recs = run_batch(rows[:1], ctx_metrics, ctx_names, [QUESTION_LEVEL_KEY])
                if recs is None:
                    ok = False
                else:
                    scores.put_many(recs)
            if ledger.stopped:
                print(f"[{i}/{n}] qid={qid}: STOPPED - {ledger.stop_reason}")
                break
            if ok:
                done_q += 1
                consecutive_failures = 0
            else:
                failed_q += 1
                consecutive_failures += 1
                failures.append({"question_id": int(qid), "errors": BATCH["errors"][:3]})
            print(f"[{i}/{n}] qid={qid} {'ok' if ok else 'transport failure (will retry next run)'}"
                  f"  spent ${ledger.spent:.4f}  elapsed {time.time() - t_start:.0f}s", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILED_QUESTIONS:
                ledger.stop(f"{consecutive_failures} consecutive questions failed in transport; "
                            "the endpoint looks unavailable. Rerun later (nothing is lost).")
                print("STOPPED -", ledger.stop_reason)
                break
    except KeyboardInterrupt:
        ledger.stop("interrupted by user")
        print("\nInterrupted. Finished questions are saved.")

    key_after = openrouter_key_info(api_key)
    summary = write_outputs(args, df, order, n, scores, ledger, cache, done_q, skipped_q,
                            failed_q, failures, ragas_version, p_in, p_out, price_desc,
                            key_before, key_after)
    print_summary(summary)
    if ledger.fatal:
        sys.exit(2)
    if ledger.stopped and ledger.stop_reason.startswith("budget"):
        sys.exit(3)


def write_outputs(args, df, order, n, scores, ledger, cache, done_q, skipped_q, failed_q,
                  failures, ragas_version, p_in, p_out, price_desc, key_before, key_after):
    sample = order.head(n)
    main_names = REJUDGED_METRICS + EMBEDDING_ONLY_METRICS
    # Question x model table (questions with every main score present).
    rows = []
    for qid in sample["question_id"]:
        for mk in MODEL_KEYS:
            if all(scores.has(qid, mk, m) for m in main_names):
                rec = {"question_id": int(qid), "model_key": mk}
                for m in main_names:
                    rec[m] = scores.data[(int(qid), mk, m)]
                rows.append(rec)
    per_q = pd.DataFrame(rows)
    meta = df[["question_id", "model_key", "is_answerable"] + main_names].rename(
        columns={m: f"{m}_original" for m in main_names})
    if len(per_q):
        per_q = per_q.merge(meta, on=["question_id", "model_key"], how="left")
        per_q = per_q.merge(sample[["question_id", "rank"]], on="question_id", how="left")
        per_q = per_q.sort_values(["rank", "model_key"])
    per_q.to_csv(os.path.join(args.out_dir, "second_judge_per_question.csv"),
                 index=False, encoding="utf-8-sig")

    ctx_rows = []
    for qid in sample["question_id"]:
        if all(scores.has(qid, QUESTION_LEVEL_KEY, m) for m in CONTEXT_METRICS):
            ctx_rows.append({"question_id": int(qid),
                             **{m: scores.data[(int(qid), QUESTION_LEVEL_KEY, m)]
                                for m in CONTEXT_METRICS}})
    pd.DataFrame(ctx_rows).to_csv(os.path.join(args.out_dir, "second_judge_context_once.csv"),
                                  index=False, encoding="utf-8-sig")

    recs = pd.DataFrame(read_jsonl(ledger.path))
    all_billed = recs if len(recs) else pd.DataFrame(
        columns=["phase", "kind", "prompt_tokens", "completion_tokens", "cost_usd", "status"])
    run_billed = pd.DataFrame(ledger.run_records) if ledger.run_records else all_billed.head(0)

    n_q_main = per_q["question_id"].nunique() if len(per_q) else 0
    n_q_ctx = len(ctx_rows)
    # Measured cost per question (all billed requests in this directory).
    by_phase_cost = float(all_billed["cost_usd"].sum()) if len(all_billed) else 0.0
    cost_per_q = by_phase_cost / n_q_main if n_q_main else float("nan")

    kinds = {}
    if len(all_billed):
        for k, g in all_billed.groupby("kind"):
            kinds[k] = {"calls": int(len(g)),
                        "prompt_tokens_mean": float(g["prompt_tokens"].mean()),
                        "completion_tokens_mean": float(g["completion_tokens"].mean()),
                        "cost_usd": float(g["cost_usd"].sum())}

    nan_counts = {}
    if len(per_q):
        for m in REJUDGED_METRICS:
            nan_counts[m] = {"second_judge_nan": int(per_q[m].isna().sum()),
                             "original_judge_nan": int(per_q[f"{m}_original"].isna().sum()),
                             "items": int(len(per_q))}
    as_check = None
    if len(per_q):
        d = (per_q["answer_similarity"] - per_q["answer_similarity_original"]).abs()
        as_check = float(d.max())

    extrap = {}
    if n_q_main and not math.isnan(cost_per_q):
        for k in EXTRAPOLATE_N:
            extrap[str(k)] = {"total_usd": cost_per_q * k,
                              "remaining_usd": max(0.0, cost_per_q * (k - n_q_main))}

    summary = {
        "mode": args.mode, "timestamp": utc_now(), "judge": args.judge,
        "original_judge": ORIGINAL_JUDGE,
        "provider_routing": provider_routing(args.provider, args.quantization),
        "providers_seen": sorted({str(p) for p in all_billed.get("provider", pd.Series(dtype=str)).dropna()}),
        "models_seen": sorted({str(p) for p in all_billed.get("model", pd.Series(dtype=str)).dropna()}),
        "prices_usd_per_million": {"in": p_in, "out": p_out, "source": price_desc},
        "ragas_version": ragas_version, "sample_seed": SAMPLE_SEED,
        "sample_size_requested": n,
        "questions_complete_main": n_q_main, "questions_complete_context": n_q_ctx,
        "questions_scored_this_run": done_q, "questions_skipped_cached": skipped_q,
        "questions_transport_failed": failed_q, "transport_failures": failures[:20],
        "requests_this_run": int(len(run_billed)), "cache_hits_this_run": cache.hits,
        "tokens_this_run": {"prompt": int(run_billed["prompt_tokens"].sum()) if len(run_billed) else 0,
                            "completion": int(run_billed["completion_tokens"].sum()) if len(run_billed) else 0},
        "cost_this_run_usd": float(run_billed["cost_usd"].sum()) if len(run_billed) else 0.0,
        "cost_cumulative_usd": float(by_phase_cost),
        "budget_usd": ledger.cap, "hard_cap_usd": HARD_BUDGET_CAP_USD,
        "stopped": ledger.stopped, "stop_reason": ledger.stop_reason,
        "length_limit_responses": int((all_billed["status"] == "length").sum()) if len(all_billed) else 0,
        "measured_cost_per_question_usd": cost_per_q,
        "per_call_kind": kinds, "extrapolation": extrap,
        "nan_counts": nan_counts,
        "answer_similarity_max_abs_diff_vs_archive": as_check,
        "openrouter_key_usage_delta_usd": (
            float(key_after["usage"]) - float(key_before["usage"])
            if key_before and key_after and key_before.get("usage") is not None
            and key_after.get("usage") is not None else None),
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def print_summary(s):
    print("\n" + "=" * 30 + " SECOND-JUDGE SUMMARY (paste this back) " + "=" * 30)
    print(f"mode={s['mode']}  judge={s['judge']}  ragas={s['ragas_version']}  "
          f"providers={s['providers_seen']}  models={s['models_seen']}")
    print(f"questions complete: main={s['questions_complete_main']}/{s['sample_size_requested']}"
          f"  context={s['questions_complete_context']}  this run={s['questions_scored_this_run']}"
          f"  cached={s['questions_skipped_cached']}  transport-failed={s['questions_transport_failed']}")
    print(f"requests this run={s['requests_this_run']}  cache hits={s['cache_hits_this_run']}  "
          f"tokens in/out={s['tokens_this_run']['prompt']:,}/{s['tokens_this_run']['completion']:,}")
    print(f"cost this run=${s['cost_this_run_usd']:.4f}  cumulative=${s['cost_cumulative_usd']:.4f}  "
          f"budget=${s['budget_usd']:.2f}  stopped={s['stopped']} {s['stop_reason']}")
    if s["openrouter_key_usage_delta_usd"] is not None:
        print(f"OpenRouter key usage delta (server side) = ${s['openrouter_key_usage_delta_usd']:.4f}")
    print(f"measured cost per question (3 models{' + context once' if s['questions_complete_context'] else ''})"
          f" = ${s['measured_cost_per_question_usd']:.5f}")
    for k, v in s["per_call_kind"].items():
        print(f"   {k:32s} calls={v['calls']:5d}  in~{v['prompt_tokens_mean']:7.0f}  "
              f"out~{v['completion_tokens_mean']:6.0f}  ${v['cost_usd']:.4f}")
    if s["extrapolation"]:
        print("extrapolation: " + "  ".join(
            f"n={k}: ${v['total_usd']:.2f} (remaining ${v['remaining_usd']:.2f})"
            for k, v in s["extrapolation"].items()))
    for m, v in s["nan_counts"].items():
        print(f"NaN {m:20s} second={v['second_judge_nan']}/{v['items']}  "
              f"original={v['original_judge_nan']}/{v['items']}")
    print(f"length-limit responses={s['length_limit_responses']}  "
          f"answer_similarity max |diff| vs archive={s['answer_similarity_max_abs_diff_vs_archive']}")
    print("=" * 100)


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["estimate", "pilot", "run"])
    ap.add_argument("--per-question-csv", default=PER_QUESTION_CSV)
    ap.add_argument("--out-dir", default=None,
                    help="default: arabicaqa_rag_results/second_judge/<judge>")
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--provider", default=JUDGE_PROVIDER,
                    help="OpenRouter provider slug to pin ('' = let OpenRouter route)")
    ap.add_argument("--quantization", default=JUDGE_QUANTIZATION)
    ap.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="cumulative spend cap for --out-dir (required for pilot/run)")
    ap.add_argument("--context-metrics", choices=["once", "skip"], default="once")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=0,
                    help="client-side request rate limit; use 18 for :free models")
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--estimate-questions", type=int, default=30)
    ap.add_argument("--tokenizer-factor", type=float, default=1.3)
    ap.add_argument("--compare-judges", default=",".join([
        JUDGE_MODEL, "qwen/qwen3-30b-a3b-instruct-2507", "deepseek/deepseek-v4-flash",
        "google/gemma-4-31b-it:free"]))
    ap.add_argument("--reasoning-off", action="store_true",
                    help="disable the thinking mode of reasoning-capable judges")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    args = ap.parse_args()

    if args.n_questions % 2:
        raise SystemExit("--n-questions must be even (half answerable, half unanswerable).")
    if args.out_dir is None:
        args.out_dir = os.path.join(OUT_ROOT, judge_slug(args.judge))
    os.makedirs(args.out_dir, exist_ok=True)
    if args.judge.endswith(":free") and not args.rpm:
        args.rpm = 18

    df = load_archive(args.per_question_csv)
    order = sample_order(df, SAMPLE_SEED)
    order.to_csv(os.path.join(args.out_dir, "sample_questions.csv"), index=False,
                 encoding="utf-8-sig")

    if args.mode == "estimate":
        estimate(args, df, order)
        return
    if args.budget_usd is None:
        raise SystemExit("--budget-usd is required for pilot and run.")
    run(args, df, order)


if __name__ == "__main__":
    main()
