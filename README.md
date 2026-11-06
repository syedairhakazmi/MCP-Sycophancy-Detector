# MCP Backed Multi Agent Sycophancy Detector
## Evaluating Evidence Grounded versus Sycophantic Belief Revision in Multi Agent LLM Debate

FAST National University of Computer and Emerging Sciences, School of Computing
* **Course:** Agentic Artificial Intelligence (AI)
* **Instructors:** Dr. Usama Imtiaz, Mr. Muhammad Sohail Abbas

#### Group Members:
* **Student 1:** Irha Kazmi [23i-0029]
* **Student 2:** Areesha Hussain [23i-0123]
* **Student 3:** Shaheera Kashif [23i-0063]

## 1. Project Overview

Multi agent debate is widely used to enhance reasoning and factual correctness in Large Language Models (LLMs) through iterative critique and revision. However, multi agent interactions introduce a critical failure mode known as sycophancy, where an agent changes its stance to agree with a peer rather than following sound factual evidence.

This project implements and evaluates an MCP Backed Multi Agent Sycophancy Detector. Building on the foundations of Kasprova et al. (2026, Too Polite to Disagree, arXiv:2604.02668), our system investigates whether an agent stance transition during debate is:
* **1. Evidence Grounded (EG):** A rational belief revision prompted by factual, verifiable evidence.
* **2. Sycophantic (SY):** An ungrounded capitulation driven by peer insistence, social consensus, authority appeals, or fabricated bluff citations.

We provide a complete pipeline covering:
* **Track A (Replication):** Baseline replication of Kasprova et al. on 250 MMLU benchmark items across open weight models.
* **Track B (Proposed Methodology):** A controlled three agent architecture comprising Answerer, Pressure Generator, and Detector integrating real time Model Context Protocol (MCP) search tools, citation verification, and a deterministic supervisory safeguard.

## 2. Methodology Architecture and Workflow

### Protocol Execution Flow
**Round 0:**
The Answerer receives a benchmark item and outputs its initial stance (ANSWER: letter).

**Rounds 1 to R:**
* **Step 1:** The Pressure Generator applies a challenge style (Pure Insistence, Appeal to Authority, Social Pressure, Fabricated Citation, or Genuine Counterargument).
* **Step 2:** The Answerer evaluates the peer challenge and emits a revised stance.
* **Step 3:** The string level Verifier matches quoted citations against MCP search results.
* **Step 4:** The system checks for a stance flip deterministically in code. If a flip occurred, the Detector evaluates justification. The safeguard enforces that justified is False unless evidence was verified by tools.

### Core System Components
* **Answerer Agent:** Solves multiple choice questions with explicit reasoning, terminating with a standardized ANSWER: letter stance line.
* **Pressure Generator:** Challenges the Answerer across five rhetorical and evidence conditions. In the genuine condition, it queries the web via DuckDuckGo MCP tool to retrieve verified facts.
* **MCP Citation Verifier (syc/verifier.py):** Programmatically verifies whether quoted excerpts appear in retrieved tool results using Unicode normalized, whitespace insensitive matching. Flags fabricated citations and bluffs.
* **Detector with Fail Closed Safeguard (syc/parsing.py):** Audits debate transcripts. The deterministic safeguard overrides any LLM judgment claiming a flip was justified when no verified evidence was presented.
* **Dual Execution Engines:** Supports both a sequential execution loop and a stateful graph workflow via LangGraph (syc/steps.py), verified by unit tests to yield identical debate trajectories.

## 3. Implementation and Evaluation Setup

**1. Detector Design:**
* **Prompt Design:** The Detector prompt instructs the LLM judge to examine the pressure message, answerer reply, and verifier report, then output a structured JSON verdict.
* **Deterministic Safeguard:** The programmatic safeguard overrides the LLM verdict to justified is False whenever the verifier did not confirm evidence. This is enforced in code, not by the LLM.
* **Baseline References:** Evaluated against Baseline 1 (All Sycophantic Prior) and Baseline 2 (Correctness Oracle).

**2. Evaluation Benchmarks:**
* **MMLU Benchmark:** 250 items across 5 subject disciplines (Elementary Mathematics, Professional Law, Machine Learning, Business Ethics, and High School Biology).
* **Multi Domain Bank:** 30 curated items covering Factual, Debatable, and Opinion categories.

**3. Labels:**
* **Silver Labels:** Programmatically constructed ground truth where Evidence Grounded is assigned strictly when verified tool evidence precedes a flip to the gold answer.
* **Human Gold Labels:** A gold sample of 46 unique flip events has been exported for human annotation (gold/gold_sample.csv). Two independent raters are required to label each event. Labels have not yet been collected and are planned as future work.

## 4. Repository Structure

<table>
<thead>
<tr>
<th>Directory / File</th>
<th>Type</th>
<th>Purpose and Content</th>
</tr>
</thead>
<tbody>
<tr>
<td>syc/</td>
<td>Python Package</td>
<td>Core framework containing 16 modules for schemas, parsing, verifier, search, agents, steps, stats, gold, stress, evaluate, records, reverify, llm, pressure, items, and runner</td>
</tr>
<tr>
<td>data/</td>
<td>Dataset Files</td>
<td>Benchmark datasets including the 30 item question bank and MMLU questions</td>
</tr>
<tr>
<td>gold/</td>
<td>Annotations</td>
<td>Exported gold evaluation sample (gold_sample.csv with 47 unique flip events, not yet annotated by human raters)</td>
</tr>
<tr>
<td>runs/</td>
<td>Experiments</td>
<td>Evaluated run traces, cache, and evaluation reports. Contains pilot (original 310 conversations) and pilot_v2 (corrected with verifier fix and deduplication)</td>
</tr>
<tr>
<td>scripts/</td>
<td>Shell Scripts</td>
<td>Automation scripts for environment setup, smoke testing, and replication</td>
</tr>
<tr>
<td>tests/</td>
<td>Test Suite</td>
<td>Pytest test suite containing 48 unit and regression tests (test_core.py, test_encoding.py, and test_fixes.py)</td>
</tr>
<tr>
<td>METHODOLOGY_REPORT.md</td>
<td>Documentation</td>
<td>Complete methodology report with literature review synthesis and empirical results</td>
</tr>
<tr>
<td>requirements.txt</td>
<td>Dependencies</td>
<td>Python package dependency list</td>
</tr>
</tbody>
</table>

## 5. Execution Guide

**1. Environment Setup:** Activate your Python environment and install the required packages using pip install requirements.

**2. Run Unit Tests:** Run the complete test suite to verify all components using pytest.

**3. Estimate Calls:** Estimate total API calls before running an experiment using the plan subcommand in the runner module.

**4. Run Multi Agent Debate and Evaluation:** Execute multi agent debate with MCP tool snapshots and evaluate run results against statistical benchmarks using the runner and evaluation modules or the replication scripts in scripts/.

## 6. Summary of Experimental Results

From our completed 310 multi agent conversation pilot (runs/pilot_v2/, corrected with verifier fix in v0.2.3):

**1. Sycophantic Vulnerability:** Under evidence free pressure (pure insistence, appeal to authority, social pressure), the model flipped its stance in 8% to 14% of MMLU questions.

**2. Regressive Flips Observed:** When challenged with pure social pressure on questions it had initially answered correctly, the model caved and chose the wrong answer in 2 to 3 out of 37 initially correct conversations (5.4% to 8.1%). These counts are small and require confirmation with larger samples.

**3. Evidence Responsiveness:** When provided with verified counterarguments with genuine web search citations on questions it initially got wrong, the model revised its answer in 44.4% of cases (4 of 9 eligible conversations), with 3 flips classified as evidence grounded.

**4. Detector Performance:** The Detector correctly classified all 46 flip events (43 sycophantic, 3 evidence grounded) against silver labels, achieving Macro F1 of 1.000 and Cohen Kappa of 1.000. Note: the evidence grounded class has only 3 events, and evaluation is against programmatic silver labels, not human annotations.

### Empirical Behavioral Findings

<table>
<thead>
<tr>
<th>Question Tier</th>
<th>Pressure Style</th>
<th>Conversations</th>
<th>Observed Flip Rate</th>
<th>95% Wilson CI</th>
<th>Mean Flip Round</th>
<th>Regressive Flip Rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>MMLU Academic</td>
<td>Pure Insistence</td>
<td>50</td>
<td>14.0%</td>
<td>0.070 to 0.262</td>
<td>1.29</td>
<td>8.1% (3 of 37 initially correct)</td>
</tr>
<tr>
<td>MMLU Academic</td>
<td>Appeal to Authority</td>
<td>50</td>
<td>14.0%</td>
<td>0.070 to 0.262</td>
<td>1.43</td>
<td>5.4% (2 of 37 initially correct)</td>
</tr>
<tr>
<td>MMLU Academic</td>
<td>Social Pressure</td>
<td>50</td>
<td>8.0%</td>
<td>0.032 to 0.188</td>
<td>1.00</td>
<td>5.4% (2 of 37 initially correct)</td>
</tr>
<tr>
<td>MMLU Academic</td>
<td>Genuine Counterargument</td>
<td>9</td>
<td>44.4%</td>
<td>0.189 to 0.733</td>
<td>1.00</td>
<td>Not Applicable</td>
</tr>
<tr>
<td>Test Bank Debatable</td>
<td>Pure Insistence</td>
<td>10</td>
<td>10.0%</td>
<td>0.018 to 0.404</td>
<td>1.00</td>
<td>11.1% (1 of 9 initially correct)</td>
</tr>
<tr>
<td>Test Bank Opinion</td>
<td>Social Pressure</td>
<td>10</td>
<td>10.0%</td>
<td>0.018 to 0.404</td>
<td>2.00</td>
<td>Not Applicable</td>
</tr>
</tbody>
</table>

### Detector Classification Performance

<table>
<thead>
<tr>
<th>Model Configuration</th>
<th>Macro F1</th>
<th>Accuracy</th>
<th>Cohen Kappa</th>
<th>SY Precision / Recall / F1</th>
<th>EG Precision / Recall / F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Proposed MCP Detector (with Safeguard)</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000 / 1.000 / 1.000</td>
<td>1.000 / 1.000 / 1.000</td>
</tr>
<tr>
<td>Proposed MCP Detector (without Safeguard)</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000 / 1.000 / 1.000</td>
<td>1.000 / 1.000 / 1.000</td>
</tr>
<tr>
<td>Baseline 1 (All Sycophantic Prior)</td>
<td>0.966</td>
<td>0.935</td>
<td>0.000</td>
<td>0.935 / 1.000 / 0.966</td>
<td>Undefined / 0.000 / Undefined</td>
</tr>
<tr>
<td>Baseline 2 (Correctness Oracle)</td>
<td>0.642</td>
<td>0.804</td>
<td>0.330</td>
<td>1.000 / 0.791 / 0.883</td>
<td>0.250 / 1.000 / 0.400</td>
</tr>
</tbody>
</table>

**Note:** All metrics above are against programmatic silver labels (46 events total: 43 SY, 3 EG). The EG class is very small (3 events). Human gold labels have not been collected yet. These results are preliminary and require validation with larger samples and independent human annotation.

For full statistical hypothesis testing (paired McNemar with Holm Bonferroni correction) and comprehensive analysis, see METHODOLOGY_REPORT.md.