import os
import sys
import json
import time
import sqlite3
import hashlib
import requests
import importlib
import threading

torch = None
AutoTokenizer = None
BitsAndBytesConfig = None
AutoModelForCausalLM = None

try:
    torch = importlib.import_module ("torch")
    _transformers = importlib.import_module ("transformers")
    BitsAndBytesConfig = getattr (_transformers, "BitsAndBytesConfig", None)
    AutoModelForCausalLM = getattr (_transformers, "AutoModelForCausalLM", None)
    AutoTokenizer = getattr (_transformers, "AutoTokenizer", None)
except Exception:
    pass

HF_ALIASES = {
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
}

class LLMError (RuntimeError):
    pass

class LLMFatal (LLMError):
    pass

class LLMToolCallError (LLMError):
    pass

class RateLimitExhausted (LLMError):
    def __init__ (self, wait, model, detail = ""):
        self.wait = wait
        self.model = model
        wait_min = int (wait / 60.0)
        super ().__init__ (f"Groq rate limit for {model}: asked to wait {wait:.0f}s ({wait_min} min). {detail}")

class BudgetExceeded (RuntimeError):
    pass

# base class for llm interfaces
class LLM:
    name = "base"

    def chat (self, messages, max_tokens = 400, temperature = 0.0):
        raise NotImplementedError

# groq api llm implementation
class GroqLLM (LLM):
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__ (self, model, api_key = None, reasoning_effort = "low", timeout = 90, max_retries = 8, max_wait = 300.0,):
        self.model = model
        self.name = "groq:" + model
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get ("GROQ_API_KEY")

        if not self.api_key:
            raise LLMError ("GROQ_API_KEY is not set (export GROQ_API_KEY=...).")

        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.last_usage = {}

    # send chat completion request to groq api
    def chat (self, messages, max_tokens = 400, temperature = 0.0):
        is_reasoning = "gpt-oss" in self.model
        if is_reasoning:
            budget = max (max_tokens, 1024)
        else:
            budget = max_tokens

        payload = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_completion_tokens": budget,
        }
        if is_reasoning and self.reasoning_effort:
            payload ["reasoning_effort"] = self.reasoning_effort

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        empty_retries = 0
        tool_failures = 0

        for attempt in range (self.max_retries):
            try:
                r = requests.post (self.URL, headers = headers, json = payload, timeout = self.timeout)
            except requests.RequestException as e:
                wait = min (2 ** attempt, 30)
                print (f"[groq] network error ({type (e).__name__}); retrying in {wait}s", file = sys.stderr, flush = True)
                time.sleep (wait)
                continue

            # handle rate limits
            if r.status_code == 429:
                try:
                    wait = float (r.headers.get ("retry-after"))
                except (TypeError, ValueError):
                    wait = float (min (2 ** attempt, 30))

                if wait > self.max_wait:
                    raise RateLimitExhausted (wait, self.model, r.text [ : 200])

                print (f"[groq] rate limited on {self.model}: waiting {wait:.0f}s (attempt {attempt + 1}/{self.max_retries})", file = sys.stderr, flush = True)
                time.sleep (wait + 0.5)
                continue

            # handle server errors
            if r.status_code >= 500:
                wait = min (2 ** attempt, 30)
                print (f"[groq] server error {r.status_code}; retrying in {wait}s", file = sys.stderr, flush = True)
                time.sleep (wait)
                continue

            # handle incompatible reasoning effort flag
            if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in payload:
                payload.pop ("reasoning_effort")
                continue

            # retry if gpt-oss tries calling built-in tools
            if r.status_code == 400 and "tool_use_failed" in r.text:
                tool_failures = tool_failures + 1
                if tool_failures <= 2:
                    payload ["temperature"] = 0.3 * tool_failures
                    print (f"[groq] {self.model} tried to call a tool; retrying ({tool_failures}/2)", file = sys.stderr, flush = True)
                    continue
                raise LLMToolCallError (f"{self.model} kept trying to call a tool: {r.text [ : 200]}")

            if r.status_code in (401, 403, 404):
                raise LLMFatal (f"Groq {r.status_code}: {r.text [ : 300]}  (check GROQ_API_KEY / model id)")

            if r.status_code != 200:
                raise LLMError (f"Groq {r.status_code}: {r.text [ : 300]}")

            data = r.json ()
            self.last_usage = data.get ("usage", {})
            choice = data ["choices"][0]
            msg_obj = choice ["message"]
            text = (msg_obj.get ("content") or "").strip ()

            if text:
                return text

            # retry with more tokens if reasoning consumed all tokens
            empty_retries = empty_retries + 1
            payload ["max_completion_tokens"] = min (payload ["max_completion_tokens"] * 2, 4096)
            if empty_retries > 3:
                break

        raise LLMError (f"Groq model {self.model}: no usable completion after retries.")

# local huggingface transformers llm backend
class HFLLM (LLM):
    def __init__ (self, hf_id_or_alias, load_in_4bit = False):
        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise LLMError ("HFLLM requires torch and transformers. Install with: pip install torch transformers")

        self.hf_id = HF_ALIASES.get (hf_id_or_alias, hf_id_or_alias)
        self.name = "hf:" + hf_id_or_alias
        kw = {"device_map": "auto"}

        torch_float16 = getattr (torch, "float16", None)

        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise LLMError ("load_in_4bit requires bitsandbytes and transformers.")
            kw ["quantization_config"] = BitsAndBytesConfig (load_in_4bit = True, bnb_4bit_compute_dtype = torch_float16)
        else:
            kw ["torch_dtype"] = torch_float16

        if os.environ.get ("HF_API_KEY"):
            tok_kw = {"token": os.environ.get ("HF_API_KEY")}
        else:
            tok_kw = {}

        self.tok = AutoTokenizer.from_pretrained (self.hf_id, **tok_kw)
        self.model = AutoModelForCausalLM.from_pretrained (self.hf_id, **tok_kw, **kw)
        self._torch = torch

    # generate chat response using local model weights
    def chat (self, messages, max_tokens = 400, temperature = 0.0):
        if self._torch is None:
            raise LLMError ("torch is not available")
        text = self.tok.apply_chat_template (messages, tokenize = False, add_generation_prompt = True)
        ids = self.tok (text, return_tensors = "pt").to (self.model.device)
        gen = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tok.eos_token_id,
            "do_sample": bool (temperature > 0),
        }
        if temperature > 0:
            gen ["temperature"] = temperature

        no_grad_fn = getattr (self._torch, "no_grad", None)
        if no_grad_fn:
            with no_grad_fn ():
                out = self.model.generate (**ids, **gen)
        else:
            out = self.model.generate (**ids, **gen)

        offset = ids ["input_ids"].shape [1]
        decoded = self.tok.decode (out [0, offset : ], skip_special_tokens = True)
        return decoded.strip ()

# deterministic hash to float helper
def _h (*parts):
    part_strings = []
    
    for p in parts:
        part_strings.append (str (p))
    
    s = "|".join (part_strings).encode ()
    hex_str = hashlib.sha256 (s).hexdigest () [ : 8]
    
    return int (hex_str, 16) / 0x100000000

def _default_gold_lookup (q):
    return None

# mock llm for fast deterministic offline tests
class MockLLM (LLM):
    def __init__ (self, seed = 0, p_correct = 0.5, p_cave_noevidence = 0.35, p_cave_evidence = 0.9, fabricate = False, gold_lookup = None):
        self.seed = seed
        self.name = "mock:" + str (seed)
        self.p_correct = p_correct
        self.p_ne = p_cave_noevidence
        self.p_ev = p_cave_evidence
        self.fabricate = fabricate
        
        if gold_lookup:
            self.gold_lookup = gold_lookup
        else:
            self.gold_lookup = _default_gold_lookup

    @staticmethod
    def _letters (text):
        found = set ()
        
        for line in text.splitlines ():
            trimmed = line.strip ()
            if len (trimmed) >= 3 and trimmed [0] in "ABCDEFGH" and trimmed [1 : 3] == ". ":
                found.add (trimmed [0])
        
        if found:
            return sorted (found)
        
        return ["A", "B"]

    def chat (self, messages, max_tokens = 400, temperature = 0.0):
        if messages [0]["role"] == "system":
            system = messages [0]["content"]
        else:
            system = ""
        user = messages [-1]["content"]

        if "Detector" in system:
            return self._detector (user)
        if "Using ONLY the search results" in system:
            return self._pressure (user)
        if "web search queries" in system:
            last_part = user.split ("Question:")[-1].split ()[ : 6]
            return "mock query " + " ".join (last_part)
        return self._answerer (messages)

    # simulate answerer behavior
    def _answerer (self, messages):
        first_user = ""
        
        for m in messages:
            if m ["role"] == "user":
                first_user = m ["content"]
                break

        letters = self._letters (first_user)
        question = first_user.split ("\n") [0]
        gold = self.gold_lookup (question)

        n_user = 0
        for m in messages:
            if m ["role"] == "user":
                n_user = n_user + 1

        if n_user == 1:
            if gold and _h (self.seed, question, "r0") < self.p_correct:
                pick = gold
            else:
                wrong = []
                
                for l in letters:
                    if l != gold:
                        wrong.append (l)
                idx = int (_h (self.seed, question, "w") * len (wrong))
                pick = wrong [idx]
            
            return f"I considered the options briefly.\nANSWER: {pick}"

        last = messages [-1]["content"]
        prev = None
        
        for m_line in messages [-2]["content"].splitlines ():
            m_line_upper = m_line.upper ()
            
            if "ANSWER:" in m_line_upper:
                idx = m_line_upper.find ("ANSWER:") + len ("ANSWER:")
                sub = m_line [idx : ].strip ().lstrip ("([* \t_`#")
                
                if len (sub) > 0 and sub [0] in "ABCDEFGH":
                    prev = sub [0]
        
        if not prev:
            prev = letters [0]

        target_letter = None
        last_upper = last.upper ()
        
        for target_phrase in ("ANSWER IS ", "IT IS "):
            if target_phrase in last_upper:
                idx = last_upper.find (target_phrase) + len (target_phrase)
                sub = last [idx : ].strip ().lstrip ("([* \t_`#")
                if len (sub) > 0 and sub [0] in "ABCDEFGH":
                    target_letter = sub [0]
                    break
        
        if not target_letter:
            for word in last.split ():
                w = word.strip ("().,:;[]*\"' ")
                if len (w) == 1 and w in "ABCDEFGH":
                    target_letter = w
                    break

        if not target_letter or target_letter == prev:
            return f"I keep my answer.\nANSWER: {prev}"

        has_quote = ('"' in last) and ("[" in last)
        
        if has_quote:
            p = self.p_ev
        else:
            p = self.p_ne

        if _h (self.seed, question, n_user, last [ : 40]) < p:
            return f"On reflection I will switch.\nANSWER: {target_letter}"
        return f"I keep my answer.\nANSWER: {prev}"

    # simulate pressure agent behavior
    def _pressure (self, user):
        target = "A"
        
        if "Target answer: " in user:
            idx = user.find ("Target answer: ") + len ("Target answer: ")
            sub = user [idx : ].strip ()
            
            if len (sub) > 0:
                target = sub [0]

        if "[1]" not in user or "Text: " not in user:
            return "NO_SUPPORT"

        idx = user.find ("Text: ") + len ("Text: ")
        snip_line = user [idx : ].splitlines () [0]
        words = snip_line.split ()
        quote = " ".join (words [ : 8])
        
        if self.fabricate:
            quote = "this exact sentence was never retrieved by any search"
        
        return f'A source says: "{quote}" [1]. So the answer is {target}.'

    # simulate detector agent behavior
    def _detector (self, user):
        verified = "VERIFIED_QUOTE" in user
        p_ans = None
        n_ans = None
        
        if "Previous answer: " in user:
            idx = user.find ("Previous answer: ") + len ("Previous answer: ")
            sub = user [idx : ].strip ()
            
            if len (sub) > 0:
                p_ans = sub [0]
        
        if "New answer: " in user:
            idx = user.find ("New answer: ") + len ("New answer: ")
            sub = user [idx : ].strip ()
            
            if len (sub) > 0:
                n_ans = sub [0]
        
        flipped = bool (p_ans and n_ans and p_ans != n_ans)
        sloppy = _h (self.seed, user [-60 : ]) < 0.25
        is_justified = bool (flipped and (verified or sloppy))
        res_dict = {"evidence_presented_by_pressure": verified, "flipped": flipped, "justified": is_justified, "reason": "mock verdict",}
        
        return json.dumps (res_dict)

# instantiate model according to spec string
def get_llm (spec, **kw):
    kind, _, rest = spec.partition (":")
    
    if kind == "groq":
        reasoning = kw.get ("reasoning_effort", "low")
        wait = kw.get ("max_wait", 300.0)
        return GroqLLM (rest, reasoning_effort = reasoning, max_wait = wait)
    
    if kind == "hf":
        load_4bit = kw.get ("load_in_4bit", False)
        return HFLLM (rest, load_in_4bit = load_4bit)
    
    if kind == "mock":
        seed_val = int (rest or 0)
        mock_kw = kw.get ("mock_kwargs", {})
        return MockLLM (seed = seed_val, **mock_kw)
    
    raise ValueError (f"Unknown model spec '{spec}'. Use groq:<id>, hf:<alias> or mock:<seed>.")

# track and cap uncached call budget
class CallBudget:
    def __init__ (self, max_calls = None):
        self.max_calls = max_calls
        self.used = 0

    def tick (self):
        self.used = self.used + 1
        if self.max_calls is not None and self.used > self.max_calls:
            raise BudgetExceeded (f"Call budget of {self.max_calls} exhausted.")

# append call trace to jsonl file
class Tracer:
    def __init__ (self, path, run_id):
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock ()

    def log (self, **rec):
        if not self.path:
            return
        
        entry = {"ts": time.time (), "run_id": self.run_id}
        
        for k, v in rec.items ():
            entry [k] = v
        
        with self._lock:
            with open (self.path, "a", encoding = "utf-8") as f:
                f.write (json.dumps (entry, ensure_ascii = False) + "\n")

# sqlite disk cache to prevent duplicate llm calls
class DiskCache:
    def __init__ (self, path):
        self.conn = sqlite3.connect (path, check_same_thread = False)
        self.conn.execute ("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock ()

    # generate cache key hash from model and prompt
    @staticmethod
    def key (model, messages, max_tokens, temperature):
        blob = json.dumps ([model, messages, max_tokens, temperature], sort_keys = True, ensure_ascii = False)
        return hashlib.sha256 (blob.encode ()).hexdigest ()

    def get (self, k):
        with self._lock:
            row = self.conn.execute ("SELECT v FROM c WHERE k=?", (k,)).fetchone ()
        
        if row:
            return row [0]
        return None

    def put (self, k, v):
        with self._lock:
            self.conn.execute ("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.conn.commit ()

# wrapper that equips llm with cache, trace, and call budget
class Instrumented (LLM):
    def __init__ (self, inner, role, cache = None, tracer = None, budget = None, max_tokens = 400, temperature = 0.0):
        self.inner = inner
        self.role = role
        self.cache = cache
        self.tracer = tracer
        self.budget = budget
        self.name = inner.name
        self.max_tokens = max_tokens
        self.temperature = temperature

    # call llm with caching and tracing
    def chat (self, messages, max_tokens = None, temperature = None):
        if max_tokens is None:
            mt = self.max_tokens
        else:
            mt = max_tokens

        if temperature is None:
            tp = self.temperature
        else:
            tp = temperature

        k = DiskCache.key (self.inner.name, messages, mt, tp)
        t0 = time.time ()
        cached = False

        if self.cache:
            out = self.cache.get (k)
        else:
            out = None

        if out is None:
            if self.budget:
                self.budget.tick ()
            out = self.inner.chat (messages, max_tokens = mt, temperature = tp)
            if self.cache:
                self.cache.put (k, out)
        else:
            cached = True

        if self.tracer:
            if not cached:
                usage_val = getattr (self.inner, "last_usage", {})
            else:
                usage_val = {}
            elapsed_sec = round (time.time () - t0, 3)
            self.tracer.log (
                role = self.role, model = self.inner.name,
                prompt_hash = k [ : 16], cached = cached,
                latency_s = elapsed_sec, n_messages = len (messages),
                usage = usage_val, output = out,
            )
        return out