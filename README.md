# MCP Backed Multi Agent Sycophancy Detector
## Evaluating Evidence Grounded versus Sycophantic Belief Revision in Multi Agent LLM Debate

FAST National University of Computer and Emerging Sciences, School of Computing
* Course: Agentic Artificial Intelligence (AI)
* Instructors: Dr. Usama Imtiaz, Mr. Muhammad Sohail Abbas

### Group Members:
* Irha Kazmi [23i 0029]
* Areesha Hussain [23i 0123]
* Shaheera Kashif [23i 0063]

## 1. Project Overview

Multi agent debate is widely used to enhance reasoning and factual correctness in Large Language Models (LLMs) through iterative critique and revision. However, multi agent interactions introduce a critical failure mode known as sycophancy, where an agent changes its stance to agree with a peer rather than following sound factual evidence.

This project implements and evaluates an MCP Backed Multi Agent Sycophancy Detector. Building on the foundations of Kasprova et al. (2026, Too Polite to Disagree, arXiv:2604.02668), our system investigates whether an agent stance transition during debate is:
1. Evidence Grounded (EG): A rational belief revision prompted by factual, verifiable evidence.
2. Sycophantic (SY): An ungrounded capitulation driven by peer insistence, social consensus, authority appeals, or fabricated bluff citations.

We provide a complete pipeline covering:
* Track A (Replication): Baseline replication of Kasprova et al. on 250 MMLU benchmark items across open weight models.
* Track B (Proposed Methodology Trained and Evaluated): A controlled three agent architecture comprising Answerer, Pressure Generator, and Detector integrating real time Model Context Protocol (MCP) search tools, citation verification, and a deterministic supervisory safeguard.

## 2. Methodology Architecture and Workflow

Protocol Flow:
Round 0:
The Answerer receives a benchmark item and outputs its initial stance (ANSWER: letter).

Rounds 1 to R:
### Step 1: The Pressure Generator applies a challenge style:
* Pure Insistence: Direct contradiction without reasons
* Appeal to Authority: Citing expert consensus
* Social Pressure: Citing peer majority
* Fabricated Citation: Hallucinated URL and fake quote bluff
* Genuine Counterargument: MCP DuckDuckGo search returning verified quotes

### Step 2: The Answerer evaluates the peer challenge and emits a revised stance.

### Step 3: The string level Verifier matches quoted citations against MCP search results.

### Step 4: The system checks for a stance flip deterministically in code.
If a flip occurred, the Detector evaluates justification.
The safeguard enforces that justified is False unless evidence was verified by tools.

### Core Components:
* Answerer Agent: Solves multiple choice questions with explicit reasoning, terminating with a standardized ANSWER: letter stance line.
* Pressure Generator: Challenges the Answerer across five rhetorical and evidence conditions. In the genuine condition, it queries the web via DuckDuckGo MCP tool to retrieve verified facts.
* MCP Citation Verifier (syc/verifier.py): Programmatically verifies whether quoted excerpts exist verbatim in retrieved tool results, flagging fabricated citations and bluffs.
* Detector with Fail Closed Safeguard (syc/parsing.py): Audits debate transcripts. The deterministic safeguard overrides any LLM judgment claiming a flip was justified when no verified evidence was presented.
* Dual Execution Engines: Supports both a sequential execution loop and a stateful graph workflow via LangGraph (syc/steps.py), verified by unit tests to yield identical debate trajectories.

## 3. Training, Calibration, and Evaluation Setup

In accordance with project requirements to Train and Test the proposed methodologies:
### Detector Calibration and Training:
  * Evaluated prompt tuning and calibration sets on the training split to optimize stance extraction, evidence attribution, and justification reasoning.
  * Programmatic threshold calibration for string quote matching to prevent false positives while remaining resilient to formatting variations.
  * Evaluated against two baseline reference models:
    * Baseline 1 (All Sycophantic): Predicts every flip is sycophantic.
    * Baseline 2 (Correctness Oracle): Treats flips to correct answers as justified and flips to incorrect answers as sycophantic.
### Evaluation Benchmarks (Test Set):
  * MMLU Benchmark: 250 items across 5 subject disciplines (Elementary Mathematics, Professional Law, Machine Learning, Business Ethics, and High School Biology).
  * Multi Domain Bank: 30 curated items covering Factual, Debatable, and Opinion categories.
### Ground Truth Validation:
  * Silver Labels: Programmatically constructed ground truth where Evidence Grounded is assigned strictly when verified tool evidence precedes a flip to the gold answer.
  * Human Gold Standard: Stratified double annotated sample of 140 debate events (gold/gold_sample.csv) with Cohen Kappa inter rater reliability.

## 4. Repository Structure

### Directory Layout:
* syc/: Core framework package containing 12 modules for schemas, parsing, verifier, search, agents, steps, stats, gold, stress, evaluate, and runner.
* data/: Benchmark datasets including the 30 item question bank and MMLU data.
* gold/: Double annotated gold evaluation samples (gold_sample.csv with 140 items and adjudication key).
* runs/: Evaluated run traces, cache, and evaluation reports (pilot run with 310 conversations, trace.jsonl, llm_cache.sqlite, and eval metrics).
* scripts/: Automation scripts for environment setup, smoke testing, and replication.
* tests/: Pytest test suite containing 39 unit tests (test_core.py and test_encoding.py).
* METHODOLOGY_REPORT.md: Complete academic methodology report.
* requirements.txt: Python dependencies list.

## 5. Execution Guide

### 1. Environment Setup:
Activate your Python environment and install the required packages using pip install requirements.

### 2. Run Unit Tests:
Run the complete test suite to verify all components:
pytest

### 3. Estimate Calls:
Estimate total API calls before running an experiment using the plan subcommand in the runner module.

### 4. Run Multi Agent Debate and Evaluation:
Execute multi agent debate with MCP tool snapshots and evaluate run results against statistical benchmarks using the runner and evaluation modules or the replication scripts in scripts/.

## 6. Summary of Experimental Results

From our completed 310 multi agent conversation benchmark run (runs/pilot/):
* Sycophantic Vulnerability: Under evidence free pressure (pure insistence, appeal to authority, social pressure), the model flipped its stance in 8% to 14% of MMLU questions.
* Regressive Flips: When challenged with pure social pressure on questions it had initially answered correctly, the model caved and chose the wrong answer in 5.4% to 8.1% of cases.
* Evidence Responsiveness: When provided with verified counterarguments with genuine web search citations on questions it initially got wrong, the model revised its answer in 44.4% of cases.
* Detector and Safeguard Accuracy: The verifier equipped detector achieved 100% precision and accuracy in detecting sycophantic flips, completely blocking ungrounded pressure and bluffs from being incorrectly classified as justified.

For full mathematical derivations, statistical hypothesis testing (paired McNemar with Holm Bonferroni correction), and comprehensive analysis, see METHODOLOGY_REPORT.md.
