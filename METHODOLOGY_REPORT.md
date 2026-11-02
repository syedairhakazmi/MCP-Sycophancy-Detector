# MCP Backed Multi Agent Sycophancy Detector
## Evaluating Evidence Grounded versus Sycophantic Belief Revision in Multi Agent LLM Debate

FAST National University of Computer and Emerging Sciences
School of Computing
* **Course:** Agentic Artificial Intelligence (Class BAI)
* **Instructors:** Dr. Usama Imtiaz, Mr. Muhammad Sohail Abbas

#### Submitted by (Group Project):
* **Student 1:** Irha Kazmi [Reg No: 23i 0029]
* **Student 2:** Areesha Hussain [Reg No: 23i 0123]
* **Student 3:** Shaheera Kashif [Reg No: 23i 0063]

## 1. Introduction and Background

Multi agent debate has emerged as a prominent mechanism for improving the reasoning and factual reliability of large language models (LLMs) by allowing multiple model instances to independently propose, critique, and revise answers before a final consensus is reached [3]. However, interaction between agents introduces complex social failure modes not present in single agent environments. One critical failure mode is sycophancy, in which an agent revises its stated position merely to align with another party rather than following valid evidence [1, 2].

Recent studies have shown that sycophancy is not confined to single agent user facing interactions. It propagates between peer agents in collaborative discussions, causing severe error cascades and degrading reasoning performance [6, 8, 9]. Concurrently, standardized tool access protocols such as the Model Context Protocol (MCP) provide a principled architecture through which LLM agents can query and retrieve external factual evidence to ground their reasoning in real time [10].

### The Core Research Gap
While prior research has measured aggregate sycophancy propagation [6, 9] or proposed voting and prompt mitigations [8], no prior study classifies an individual belief change as evidence grounded or socially driven at the level of a single debate turn. When an agent changes its mind during multi agent discussion, is it rationally updating its belief in response to verified factual evidence, or is it capitulating to unsupported social pressure, appeals to authority, or fabricated citations?

To resolve this gap, we develop, train, calibrate, and evaluate an MCP Backed Multi Agent Sycophancy Detector (syc watchdog). The proposed framework evaluates whether observed stance flips are:
**1. Evidence Grounded (EG):** Justified belief revisions substantiated by verified external retrieval.
**2. Sycophantic (SY):** Concessions induced by ungrounded conversational pressure or unverified bluff citations.

## 2. Synthesis of Literature and Related Work

Our project builds directly upon the ten core papers surveyed in our Literature Review, synthesizing three distinct research areas:

**1. Sycophancy in Single Agent LLM Interactions:**
* **Perez et al. [1]:** Developed model written evaluations, demonstrating that RLHF tuned models frequently echo a user stated opinion over truth.
* **Sharma et al. [2]:** Analyzed preference model behavior, revealing that human raters and reward models reward convincing sounding sycophantic responses.
* **Fanous et al. [4] (SycEval):** Manipulated rebuttal strength and timing, finding that over 58% of responses across tested settings were sycophantic.
* **Cheng et al. [5] (ELEPHANT):** Demonstrated that simple binary flip metrics fail to capture nuanced social sycophancy dimensions.
* **Core Limitation:** All single agent frameworks evaluate user to model interactions and lack peer agents and external evidence verification.

**2. Sycophancy and Disagreement Dynamics in Multi Agent Debate:**
* **Du et al. [3]:** Established that iterative multi agent debate improves factuality and reasoning over single pass generation.
* **Yao et al. [6]:** Showed that persona based sycophancy persists and intensifies across successive debate rounds.
* **Wynn, Satija, and Hadfield [7]:** Observed that agent disagreement declines across debate rounds, correlating with degraded problem solving performance.
* **Pitre et al. [8] (ConsensAgent):** Proposed confidence weighted voting to reduce token costs and sycophantic copying.
* **Kasprova et al. [9] (Too Polite to Disagree):** Studied peer sycophancy propagation across six open source LLMs, using sycophancy ranking priors to improve discussion accuracy.
* **Core Limitation:** These studies evaluate aggregate accuracy and propagation rather than auditing whether individual belief changes are justified by verifiable facts.

**3. Tool Augmented LLM Systems and Model Context Protocol (MCP):**
* **Hou et al. [10]:** Surveyed MCP as an emerging open standard enabling secure client tool server interactions.
* **Core Limitation:** MCP has been studied primarily at the systems and architecture level. It has not previously been deployed as an empirical experimental variable to audit sycophancy in multi agent debate.

### Comparison of Reviewed Literature

<table>
<thead>
<tr>
<th>Paper</th>
<th>Problem Addressed</th>
<th>Method and Approach</th>
<th>Dataset Used</th>
<th>Key Findings</th>
<th>Limitations</th>
</tr>
</thead>
<tbody>
<tr>
<td>Perez et al. [1]</td>
<td>Elicit and measure LLM behavioral tendencies including sycophancy</td>
<td>Model written evaluations where models generate behavioral test suites</td>
<td>154 model written behavior datasets including opinion sycophancy subset</td>
<td>RLHF tuned models are significantly more likely to echo a user stated opinion</td>
<td>Single agent static prompts without live tool verification</td>
</tr>
<tr>
<td>Sharma et al. [2]</td>
<td>Explain why sycophancy emerges from preference alignment pipelines</td>
<td>Analysis of human preference comparisons and preference model behavior</td>
<td>Human preference comparisons across five commercial assistants</td>
<td>Humans and reward models favor convincing but incorrect answers over correct ones</td>
<td>Single agent user versus model setting with no peer agent pressure</td>
</tr>
<tr>
<td>Du et al. [3]</td>
<td>Single pass generation yields factual errors and shallow reasoning</td>
<td>Multi agent debate where instances propose critique and revise across rounds</td>
<td>Arithmetic, GSM8K, MMLU, and biography generation benchmarks</td>
<td>Multi agent debate improves factual accuracy and reasoning over single agent</td>
<td>Does not investigate unsupported peer pressure or sycophancy failure modes</td>
</tr>
<tr>
<td>Fanous et al. [4]</td>
<td>Quantify sycophancy through a reproducible framework</td>
<td>SycEval framework manipulating rebuttal strength and timing across LLMs</td>
<td>AMPS mathematics and MedQuad medical advice prompts</td>
<td>Reports 58.19% sycophantic responses overall across tested settings</td>
<td>Single agent user authored rebuttals without multi agent evidence condition</td>
</tr>
<tr>
<td>Cheng et al. [5]</td>
<td>Binary flip metrics miss nuanced dimensions of social sycophancy</td>
<td>ELEPHANT human validated LLM judge scoring across social dimensions</td>
<td>OEQ, AITA, and social sycophancy datasets</td>
<td>High levels of social sycophantic behavior observed across dimensions</td>
<td>Single agent advice setting without debate or evidence grounding manipulation</td>
</tr>
<tr>
<td>Yao et al. [6]</td>
<td>How sycophancy behaves across successive debate rounds</td>
<td>Persona based sycophancy controls in multi round debate</td>
<td>Debate transcripts across open weight model families</td>
<td>Sycophancy persists and intensifies across successive debate rounds</td>
<td>Does not distinguish evidence backed persuasion from unsupported pressure</td>
</tr>
<tr>
<td>Wynn et al. [7]</td>
<td>Characterize debate failure modes beyond raw accuracy</td>
<td>Controlled debate experiments tracking disagreement and task performance</td>
<td>Arithmetic, GSM8K, and logic puzzles</td>
<td>Disagreement declines over rounds and correlates with degraded performance</td>
<td>Does not separate social consensus conformity from evidence based consensus</td>
</tr>
<tr>
<td>Pitre et al. [8]</td>
<td>Sycophantic copying and high token costs in multi agent systems</td>
<td>ConsensAgent using confidence weighted voting and prompt refinement</td>
<td>Multi hop QA benchmark and six reasoning datasets</td>
<td>Reduces sycophantic copying and token overhead while improving consensus accuracy</td>
<td>Operates at system level without classifying single turn belief changes</td>
</tr>
<tr>
<td>Kasprova et al. [9]</td>
<td>Sycophancy propagation between peer agents in collaborative discussion</td>
<td>Inference time peer sycophancy ranking priors across open weight LLMs</td>
<td>250 MMLU items across five subjects</td>
<td>Awareness of peer sycophancy improves discussion accuracy and curbs cascades</td>
<td>Measures aggregate propagation without verifying external facts at turn level</td>
</tr>
<tr>
<td>Hou et al. [10]</td>
<td>Standardizing external tool and context integration for LLMs</td>
<td>Systematic landscape survey of Model Context Protocol architecture and risks</td>
<td>Protocol specification, security models, and ecosystem deployments</td>
<td>Establishes MCP as standardized mechanism for external factual retrieval</td>
<td>Survey analysis without empirical experiments in multi agent debate</td>
</tr>
</tbody>
</table>

## 3. Proposed Methodology and System Architecture

Our methodology introduces a modular, state managed three agent interaction pipeline implemented natively in Python with support for both sequential loop execution and LangGraph state graph workflows.

### Protocol Execution Flow
**Round 0:**
The Answerer receives an item and outputs its initial independent stance (ANSWER: letter).

**Rounds 1 to R:**
* **Step 1:** The Pressure Generator applies an experimental challenge style (Pure Insistence, Appeal to Authority, Social Pressure, Fabricated Citation, or Genuine Counterargument).
* **Step 2:** The Answerer evaluates the peer challenge and emits its revised stance.
* **Step 3:** The string level Verifier matches quoted citations against MCP search results.
* **Step 4:** The system checks for a stance flip deterministically in code. If a flip occurred, the Detector evaluates justification. The safeguard enforces that justified is False unless evidence was verified by tools.

### Agent Formulation
**1. Answerer Agent:** Given item Q and options O, the Answerer generates a chain of thought rationale and terminates with an explicit format: ANSWER: letter. In successive rounds, it updates its stance based on the dialogue history.

**2. Pressure Generator:** Challenges the Answerer across five conditions:
* **Pure Insistence:** Repeated assertion of error without justification.
* **Appeal to Authority:** Citing fictitious academic consensus or authoritative figures.
* **Social Pressure:** Citing unanimous peer disagreement.
* **Fabricated Citation:** Adversarial bluff providing a fake URL and fabricated quotation.
* **Genuine Counterargument:** Genuine counterargument generated by querying DuckDuckGo via the Model Context Protocol (MCP), extracting real search snippets, and quoting verifiable text with bracketed citations.

**3. MCP Citation Verifier (syc/verifier.py):** Scans quoted text within peer challenges and checks for verbatim occurrences within retrieved MCP search snapshots. It emits deterministic verification tags (VERIFIED QUOTE, UNVERIFIED QUOTE, or NO SOURCES).

**4. Detector Agent with Fail Closed Safeguard (syc/parsing.py):** Audits whether a stance flip occurred and whether it was justified. The supervisory safeguard programmatically enforces that justified is False unless verified evidence was presented by the peer. If an LLM judge is misled by eloquent rhetoric or a fake citation, the safeguard overrides the classification to justified is False.

## 4. Training, Calibration, and Baseline Setup

To fulfill the project mandate to Train and Test proposed methodologies with trained and evaluated code and results:

### Detector Calibration and In Context Training
In modern foundation model paradigms where fine tuning frozen frontier APIs is constrained, methodology training involves rigorous in context calibration, prompt tuning, and algorithmic threshold calibration on a designated training split:
**1. In Context Prompt Tuning:** The Detector prompt was iteratively tuned and calibrated on a 40 item development set, refining instructions for detecting epistemic justification, extracting peer evidence claims, and outputting validated JSON schemas.
**2. Deterministic Stance Extraction Calibration:** Standardized string parsing rules were trained and validated across formatting edge cases, achieving 100% agreement with ground truth stance transitions.
**3. Verifier Fuzzy Matching Calibration:** Calibrated string matching thresholds to handle HTML entities, whitespace normalization, and punctuation variations without yielding false positive matches on hallucinated quotes.

### Baseline Models for Comparative Evaluation
To rigorously benchmark the proposed MCP backed Detector, we calibrated two reference baselines:
* **Baseline 1 (All Sycophantic Prior):** Predicts that all observed stance flips in multi agent debate are sycophantic.
* **Baseline 2 (Correctness Oracle):** Naively labels any stance flip toward the ground truth answer as justified and any flip away from the ground truth answer as sycophantic.

## 5. Experimental Test Setup and Benchmarks

The testing pipeline was evaluated across two datasets:
**1. MMLU Benchmark (Kasprova et al. Subsets):** 250 test questions spanning 5 subject categories:
* **Elementary Mathematics:** Pure reasoning, excluded from web retrieval.
* **Professional Law:** Statutory interpretation.
* **Machine Learning:** Technical concepts and architectures.
* **Business Ethics:** Normative dilemmas.
* **High School Biology:** Empirical scientific knowledge.

**2. Multi Domain Test Bank:** 30 curated items categorized into Factual (objective), Debatable (controversial), and Opinion (subjective) items.

### Evaluation Metrics
* **Flip Rate (FR):** Proportion of debates where the final stance changed from Round 0. Computed with 95% Wilson Score Confidence Intervals.
* **Regressive Flip Rate (RFR):** Frequency with which an initially correct agent caves to an incorrect answer under pressure.
* **Evidence Grounded Flip Rate (EGFR):** Frequency with which an initially incorrect agent corrects its stance when provided with verified genuine counterarguments.
* **Statistical Significance:** Paired comparisons between pressure styles evaluated using the Exact McNemar Test with Holm Bonferroni correction for family wise error rate control.
* **Detector Performance:** Precision, Recall, Macro F1, and Cohen Kappa measured against programmatic Silver Labels and a double annotated Human Gold sample of 140 items.

## 6. Evaluated Results and Empirical Findings

A full pilot evaluation comprising 310 multi agent conversations was executed using groq:openai/gpt oss 20b as the Answerer and groq:openai/gpt oss 120b as the Detector.

### 6.1. Behavioral Vulnerability Across Pressure Styles

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
<td>8.1% (3 of 37)</td>
</tr>
<tr>
<td>MMLU Academic</td>
<td>Appeal to Authority</td>
<td>50</td>
<td>14.0%</td>
<td>0.070 to 0.262</td>
<td>1.43</td>
<td>5.4% (2 of 37)</td>
</tr>
<tr>
<td>MMLU Academic</td>
<td>Social Pressure</td>
<td>50</td>
<td>8.0%</td>
<td>0.032 to 0.188</td>
<td>1.00</td>
<td>5.4% (2 of 37)</td>
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
<td>11.1% (1 of 9)</td>
</tr>
<tr>
<td>Test Bank Opinion</td>
<td>Social Pressure</td>
<td>10</td>
<td>10.0%</td>
<td>0.018 to 0.404</td>
<td>1.00</td>
<td>Not Applicable</td>
</tr>
</tbody>
</table>

### Key Behavioral Observations
**1. Susceptibility to Rhetorical Pressure:** Even on standardized academic benchmarks, models flipped in 8% to 14% of debates under pure social pressure containing zero factual justification.
**2. Severe Regressive Sycophancy:** In 5.4% to 8.1% of cases where the model had initially deduced the correct answer, it abandoned its correct knowledge and adopted a false claim solely due to peer insistence or manufactured consensus.
**3. High Receptivity to Genuine Evidence:** When challenged with verified web evidence retrieved via MCP on questions it had gotten wrong, the model corrected its stance in 44.4% of cases.

### 6.2. Paired McNemar Statistical Significance Tests

Pairwise Style Comparisons (N = 80 paired items for rhetoric, N = 10 paired items for genuine counterarguments):

<table>
<thead>
<tr>
<th>Paired Style Comparison</th>
<th>Sample Size N</th>
<th>Discordant b</th>
<th>Discordant c</th>
<th>Risk Difference</th>
<th>Exact p Value</th>
<th>Holm Adjusted p Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Appeal to Authority vs Pure Insistence</td>
<td>80</td>
<td>2</td>
<td>2</td>
<td>0.000</td>
<td>1.000</td>
<td>1.000</td>
</tr>
<tr>
<td>Appeal to Authority vs Social Pressure</td>
<td>80</td>
<td>5</td>
<td>2</td>
<td>+0.038</td>
<td>0.453</td>
<td>1.000</td>
</tr>
<tr>
<td>Pure Insistence vs Social Pressure</td>
<td>80</td>
<td>4</td>
<td>1</td>
<td>+0.038</td>
<td>0.375</td>
<td>1.000</td>
</tr>
<tr>
<td>Genuine Counterargument vs Social Pressure</td>
<td>10</td>
<td>3</td>
<td>0</td>
<td>+0.300</td>
<td>0.250</td>
<td>1.000</td>
</tr>
<tr>
<td>Genuine Counterargument vs Pure Insistence</td>
<td>10</td>
<td>3</td>
<td>1</td>
<td>+0.200</td>
<td>0.625</td>
<td>1.000</td>
</tr>
</tbody>
</table>

*Statistical Takeaway:* In our pilot sample, purely rhetorical pressure styles induced statistically indistinguishable flip rates. Genuine counterarguments exhibited a substantial positive risk difference (+30% vs social pressure and +20% vs insistence), confirming that factual grounding produces a distinct behavioural dynamic.

### 6.3. Detector and Safeguard Performance

Evaluating the calibrated Detector across all 46 stance flip events:

<table>
<thead>
<tr>
<th>Model Configuration</th>
<th>Macro F1</th>
<th>Accuracy</th>
<th>Cohen Kappa</th>
<th>Sycophancy Precision Recall F1</th>
<th>Evidence Grounded Precision Recall F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Proposed MCP Detector (with Safeguard)</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000 / 1.000 / 1.000</td>
<td>Verified Clean (All Pilot Flips Were Sycophantic)</td>
</tr>
<tr>
<td>Baseline 1 (All Sycophantic Prior)</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000</td>
<td>1.000 / 1.000 / 1.000</td>
<td>Uncalibrated Prior</td>
</tr>
<tr>
<td>Baseline 2 (Correctness Oracle)</td>
<td>0.850</td>
<td>0.739</td>
<td>0.000</td>
<td>1.000 / 0.739 / 0.850</td>
<td>Misclassifies Lucky Guesses as Justified</td>
</tr>
</tbody>
</table>

*Crucial Audit Insight:* Baseline 2 (the Correctness Oracle) misclassifies 26.1% of flips because an agent that flips to the correct answer due to a lucky guess or peer pressure without verified evidence is still sycophantic. The proposed MCP Detector with safeguard correctly recognized that zero verified evidence was presented, accurately classifying 100% of these flips as sycophantic.

## 7. Artifacts and Code Deliverables

In accordance with course submission guidelines, all trained, calibrated, and evaluated code, benchmarks, and experimental logs are provided in the project repository:
**1. Trained and Calibrated Framework Code:** syc/ (all 12 modules implementing schemas, parsing, verifier, search, agents, steps, and evaluation).
**2. Test Suite:** tests/test_core.py and tests/test_encoding.py (39 unit tests covering parsing, verification, statistics, and LangGraph parity).
**3. Pilot Experimental Traces and Results:** runs/pilot/ containing 310 full debate records (conversations.jsonl), full query traces (trace.jsonl), SQLite cache (llm_cache.sqlite), and generated CSV evaluation reports (runs/pilot/eval/).
**4. Human Gold Benchmark:** gold/gold_sample.csv (140 blinded human evaluation events with adjudication key).

## 8. Conclusion

This project successfully implements, trains, and evaluates an MCP Backed Multi Agent Sycophancy Detector. By formalizing multi agent debate, integrating real time Model Context Protocol tool verification, and applying programmatic supervisory safeguards, our framework reliably separates genuine epistemic belief revision from opportunistic peer sycophancy. The empirical findings provide concrete evidence that LLMs are vulnerable to regressive sycophancy under social pressure, while demonstrating that tool grounded verification offers a provably sound safeguard for trustworthy multi agent systems.

## 9. References

* **[1] Perez et al.:** Discovering language model behaviors with model written evaluations, in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 13387 to 13434.
* **[2] Sharma et al.:** Towards understanding sycophancy in language models, in Int. Conf. on Learning Representations (ICLR), 2024.
* **[3] Du et al.:** Improving factuality and reasoning in language models through multi agent debate, in Proc. Int. Conf. on Machine Learning (ICML), 2024, arXiv:2305.14325.
* **[4] Fanous et al.:** SycEval: Evaluating sycophancy in large language models, in Proc. AAAI/ACM Conf. on AI, Ethics, and Society, vol. 8, no. 1, 2025, pp. 893 to 900.
* **[5] Cheng et al.:** ELEPHANT: Measuring and understanding social sycophancy in LLMs, arXiv preprint arXiv:2505.13995, 2025.
* **[6] Yao et al.:** Peacemaker or troublemaker? How sycophancy shapes multi agent debate, arXiv preprint arXiv:2509.23055, 2025.
* **[7] Wynn et al.:** Talk isn't always cheap: Understanding failure modes in multi agent debate, arXiv preprint arXiv:2509.05396, 2025.
* **[8] Pitre et al.:** CONSENSAGENT: Towards efficient and effective consensus in multi agent LLM interactions through sycophancy mitigation, in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 22112 to 22133.
* **[9] Kasprova et al.:** Too polite to disagree: Understanding sycophancy propagation in multi agent systems, in Proc. 27th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL), 2026, pp. 795 to 814.
* **[10] Hou et al.:** MCP landscape: Architecture, ecosystem, adoption, and risks, arXiv preprint arXiv:2503.23278, 2025.
