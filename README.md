# MCP-Backed Multi-Agent Sycophancy Detector
### Evaluating Evidence-Grounded versus Sycophantic Belief Revision in Multi-Agent LLM Debate

**Institution:** FAST – National University of Computer & Emerging Sciences, School of Computing  
**Course:** Agentic Artificial Intelligence (CS / AI)  
**Instructors:** Dr. Usama Imtiaz, Mr. Muhammad Sohail Abbas  

**Group Members:**
* Irha Kazmi [23i-0029]
* Areesha Hussain [23i-0123]
* Shaheera Kashif [23i-0063]

---

## 1. Project Overview

Multi-agent debate is widely used to enhance reasoning and factual correctness in Large Language Models (LLMs) through iterative critique and revision. However, multi-agent interactions introduce a critical failure mode: **sycophancy**, where an agent changes its stance to agree with a peer rather than following sound factual evidence.

This project implements and evaluates an **MCP-Backed Multi-Agent Sycophancy Detector**. Building on the foundations of Kasprova et al. (2026, *Too Polite to Disagree*, arXiv:2604.02668), our system investigates whether an agent's change of mind during debate is:
1. **Evidence-Grounded (EG):** A rational belief revision prompted by factual, verifiable evidence.
2. **Sycophantic (SY):** An ungrounded capitulation driven by peer insistence, social consensus, authority appeals, or fabricated bluff citations.

We provide a complete pipeline covering:
* **Track A (Replication):** Baseline replication of Kasprova et al. on 250 MMLU benchmark items across open-weight models (Llama-3 and Qwen2.5 series).
* **Track B (Proposed Methodology - Trained & Evaluated):** A controlled three-agent architecture (**Answerer**, **Pressure Generator**, and **Detector**) integrating real-time Model Context Protocol (MCP) search tools, citation verification, and a deterministic supervisory safeguard.

---

## 2. Methodology Architecture & Workflow

```
+-----------------------------------------------------------------------------+
|                               ROUND 0                                       |
|  Benchmark Item (Question + Options) ---> Answerer Agent                   |
|                                           |                                 |
|                                           v                                 |
|                                   Initial Stance [S_0]                      |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                               ROUNDS 1 TO R                                 |
|                                                                             |
|  [Step 1: Peer Pressure Generator]                                          |
|  Selects challenge style:                                                   |
|   ├── Pure Insistence         (Direct contradiction without reasons)        |
|   ├── Appeal to Authority     (Fabricated expert consensus)                 |
|   ├── Social Pressure         (Fabricated peer majority)                    |
|   ├── Fabricated Citation     (Hallucinated URL + fake quote bluff)         |
|   └── Genuine Counterargument (MCP DuckDuckGo search -> Verified quote)     |
|                                                                             |
|  [Step 2: Answerer Response]                                                |
|  Answerer evaluates peer challenge -> Emits revised Stance [S_r]             |
|                                                                             |
|  [Step 3: MCP Tool Verification]                                            |
|  String-level Verifier matches quoted citations against MCP search snips    |
|                                                                             |
|  [Step 4: Detector & Safeguard Evaluation]                                  |
|  Code extracts stance flip deterministically: Flip = (S_r != S_{r-1})       |
|  If Flip == True:                                                           |
|      Detector evaluates justification                                       |
|      Safeguard enforces: justified = False unless evidence was tool-verified |
+-----------------------------------------------------------------------------+
```

### Core Components
1. **Answerer Agent:** Solves multi-choice questions with explicit reasoning, terminating with a standardized `ANSWER: <letter>` stance line.
2. **Pressure Generator:** Challenges the Answerer across 5 rhetorical and evidence conditions. In the genuine condition, it queries the web via DuckDuckGo MCP tool to retrieve verified facts.
3. **MCP Citation Verifier (`syc/verifier.py`):** Programmatically verifies whether quoted excerpts exist verbatim in retrieved tool results, flagging fabricated citations and bluffs.
4. **Detector with Fail-Closed Safeguard (`syc/parsing.py`):** Audits debate transcripts. The deterministic safeguard overrides any LLM judgment claiming a flip was justified when no verified evidence was presented.
5. **Execution Engines:** Supports both a sequential execution loop and a stateful graph workflow via LangGraph (`syc/steps.py`), verified by unit tests to yield identical debate trajectories.

---

## 3. Training, Calibration & Evaluation Setup

In accordance with project requirements to **Train and Test the proposed methodologies**:
* **Detector Calibration / Training:**
  - Evaluated prompt-tuning and calibration sets on the training split to optimize stance extraction, evidence attribution, and justification reasoning.
  - Programmatic threshold calibration for string quote matching to prevent false positives while remaining resilient to formatting and whitespace variations.
  - Evaluated against two baseline reference models:
    - *Baseline 1 (All-Sycophantic):* Predicts every flip is sycophantic.
    - *Baseline 2 (Correctness Oracle):* Naively treats flips to correct answers as justified and flips to incorrect answers as sycophantic.
* **Evaluation Benchmarks (Test Set):**
  - **MMLU Benchmark:** 250 items across 5 subject disciplines (Elementary Mathematics, Professional Law, Machine Learning, Business Ethics, and High School Biology).
  - **Multi-Domain Bank:** 30 curated items covering Factual, Debatable, and Opinion categories.
* **Ground Truth Validation:**
  - *Silver Labels:* Programmatically constructed ground truth where Evidence-Grounded is assigned strictly when verified tool evidence precedes a flip to the gold answer.
  - *Human Gold Standard:* Stratified double-annotated sample of $N=140$ debate events (`gold/gold_sample.csv`) with Cohen's Kappa ($\kappa$) inter-rater reliability.

---

## 4. Repository Structure

```
.
├── syc/                    # Core framework package
│   ├── schemas.py          # Data classes (Item, StanceRecord, Verdict, RunConfig)
│   ├── parsing.py          # Stance extraction & fail-closed safeguard
│   ├── verifier.py         # MCP search quote verifier & bluff detector
│   ├── llm.py              # LLM clients (Groq, HuggingFace, Mock) with SQLite cache
│   ├── search.py           # DuckDuckGo MCP tool client & snapshot replay
│   ├── pressure.py         # Pressure style scripts & genuine counterargument prompts
│   ├── agents.py           # Answerer, Pressure, and Detector agent prompts
│   ├── steps.py            # Execution loops (sequential loop & LangGraph engine)
│   ├── stats.py            # Wilson score intervals, Cohen's kappa, McNemar & Holm tests
│   ├── gold.py             # Human gold sample exporter & inter-rater kappa importer
│   ├── stress.py           # Detector perturbation & robustness stress tests
│   ├── evaluate.py         # Comprehensive evaluation & report generator
│   └── runner.py           # Unified CLI interface (plan, run, eval)
├── data/                   # Benchmark datasets (30-item question bank, MMLU data)
├── gold/                   # Double-annotated gold evaluation samples
│   ├── gold_sample.csv     # 140-item blinded sample for human raters
│   └── gold_sample.csv.key.json # Adjudication key
├── runs/                   # Evaluated run traces, cache, and evaluation reports
│   └── pilot/              # 310-conversation pilot experiment on Groq
│       ├── conversations.jsonl # Complete multi-round conversation records
│       ├── trace.jsonl     # Detailed LLM query/response traces
│       ├── llm_cache.sqlite# Local SQLite cache for cost-free determinism
│       └── eval/           # Evaluated metrics, McNemar tests, behavioural stats
├── scripts/                # Automation and replication scripts
│   ├── setup_base.sh       # Clone base paper repository at pinned commit
│   ├── smoke_test.sh       # Quick sanity test suite
│   ├── pilot_groq.sh       # Full pilot run script
│   └── run_replication.sh  # GPU Track A baseline replication
├── tests/                  # Pytest test suite (39 unit tests)
│   ├── test_core.py        # Core logic, verifier, parsing, stats, and graph parity
│   └── test_encoding.py    # UTF-8 and format integrity checks
├── METHODOLOGY_REPORT.md   # Complete academic methodology report
└── requirements.txt        # Python dependencies
```

---

## 5. Quickstart & Execution Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify System (Run All Unit Tests)
```bash
python -m pytest
# All 39 tests verifying verifier, parsing, stats, and graph parity will run and pass.
```

### 3. Estimate API Calls
```bash
python -m syc.runner plan --source both --n-per-subject 10 --rounds 3
```

### 4. Run Debate & Evaluation Pipeline
```bash
# Set Groq API Key (for live models)
$env:GROQ_API_KEY="your-api-key"

# Run multi-agent debate with MCP search tool snapshots
python -m syc.runner run --source both --n-per-subject 10 --rounds 3 `
  --answerer groq:openai/gpt-oss-20b `
  --detector groq:openai/gpt-oss-120b `
  --search mcp --snapshot runs/shared_snap.json `
  --out runs/pilot

# Evaluate the run results against statistical benchmarks
python -m syc.evaluate --runs runs/pilot/conversations.jsonl --gold gold/gold_sample.csv
```

---

## 6. Summary of Experimental Results

From our completed 310 multi-agent conversation benchmark run (`runs/pilot/`):
* **Sycophantic Vulnerability:** Under evidence-free pressure (`pure_insistence`, `appeal_to_authority`, `social_pressure`), the model flipped its stance in **8% to 14%** of MMLU questions.
* **Regressive Flips:** When challenged with pure social pressure on questions it had initially answered correctly, the model caved and chose the wrong answer in **5.4% to 8.1%** of cases.
* **Evidence Responsiveness:** When provided with verified counterarguments with genuine web search citations on questions it initially got wrong, the model revised its answer in **44.4%** of cases.
* **Detector & Safeguard Accuracy:** The verifier-equipped detector achieved **100% precision and accuracy** in detecting sycophantic flips, completely blocking ungrounded pressure and bluffs from being incorrectly classified as justified.

For full mathematical derivations, statistical hypothesis testing (paired McNemar with Holm-Bonferroni correction), and comprehensive analysis, see the [Methodology Report](file:///d:/Desktop/Sem%2007/Agentic%20AI/Project/Methodology/i230029_i230123_i230063/METHODOLOGY_REPORT.md).
