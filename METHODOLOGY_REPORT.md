# MCP Backed Multi Agent Sycophancy Detector
## Evaluating Evidence Grounded versus Sycophantic Belief Revision in Multi Agent LLM Debate

FAST National University of Computer and Emerging Sciences, School of Computing
* **Course:** Agentic Artificial Intelligence (AI)
* **Instructors:** Dr. Usama Imtiaz, Mr. Muhammad Sohail Abbas

#### Group Members:
* **Student 1:** Irha Kazmi [23i-0029]
* **Student 2:** Areesha Hussain [23i-0123]
* **Student 3:** Shaheera Kashif [23i-0063]

## 1. Introduction and Background

Multi agent debate has emerged as a prominent mechanism for improving the reasoning and factual reliability of large language models (LLMs) by allowing multiple model instances to independently propose, critique, and revise answers before a final consensus is reached [3]. However, interaction between agents introduces complex social failure modes not present in single agent environments. One critical failure mode is sycophancy, in which an agent revises its stated position merely to align with another party rather than following valid evidence [1, 2].

Recent studies have shown that sycophancy is not confined to single agent user facing interactions. It propagates between peer agents in collaborative discussions, causing severe error cascades and degrading reasoning performance [6, 8, 9]. Concurrently, standardized tool access protocols such as the Model Context Protocol (MCP) provide a principled architecture through which LLM agents can query and retrieve external factual evidence to ground their reasoning in real time [10].

### The Core Research Gap
While prior research has measured aggregate sycophancy propagation [6, 9] or proposed voting and prompt mitigations [8], no prior study classifies an individual belief change as evidence grounded or socially driven at the level of a single debate turn. When an agent changes its mind during multi agent discussion, is it rationally updating its belief in response to verified factual evidence, or is it capitulating to unsupported social pressure, appeals to authority, or fabricated citations?

To address this gap, we develop and evaluate an MCP Backed Multi Agent Sycophancy Detector (syc watchdog). The proposed framework classifies whether observed stance flips are:
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
* **Pitre et al. [8] (ConsensAgent):** Proposed prompt refinement techniques to reduce token costs and sycophantic copying in multi agent systems.
* **Kasprova et al. [9] (Too Polite to Disagree):** Studied peer sycophancy propagation across six open source LLMs, using sycophancy ranking priors to improve discussion accuracy.
* **Core Limitation:** These studies evaluate aggregate accuracy and propagation rather than auditing whether individual belief changes are justified by verifiable facts.

**3. Tool Augmented LLM Systems and Model Context Protocol (MCP):**
* **Hou et al. [10]:** Surveyed the Model Context Protocol landscape including architecture, ecosystem, adoption, and associated risks.
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
<td>ConsensAgent using prompt refinement to reduce sycophantic copying</td>
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

**3. MCP Citation Verifier (syc/verifier.py):** Scans quoted text within peer challenges and checks for occurrence within retrieved MCP search snapshots. Matching ignores whitespace, punctuation, and case differences to handle snippets returned by search engines with stripped formatting. It emits deterministic verification tags (VERIFIED QUOTE, UNVERIFIED QUOTE, or NO SOURCES).

**4. Detector Agent with Fail Closed Safeguard (syc/parsing.py):** Audits whether a stance flip occurred and whether it was justified. The supervisory safeguard programmatically enforces that justified is False unless verified evidence was presented by the peer. If an LLM judge is misled by eloquent rhetoric or a fake citation, the safeguard overrides the classification to justified is False.

## 4. Implementation Details

### Detector Prompt Design
The Detector prompt instructs the LLM judge to examine the pressure message, the answerer reply, and (when available) the verifier report, then output a structured JSON verdict with three fields: whether the answerer flipped, whether the peer presented evidence, and whether the flip was justified. The prompt was written once and tested against the pilot data.

### Deterministic Stance Extraction
Standardized string parsing rules extract the Answerer stance from free text replies. The parser searches for an explicit ANSWER: letter pattern. These rules are validated implicitly through the test suite (48 passing tests covering multiple formatting edge cases).

### Verifier Matching
The verifier uses Unicode normalization and alphanumeric squashing to compare quoted text against search snippets. This handles the common case where search engines (such as DuckDuckGo) return snippets with spaces stripped or punctuation altered. Ellipsis markers (such as the Unicode ellipsis character or three dots) are treated as fragment separators, allowing partial quote matching. The verifier was corrected in version 0.2.3 after the initial pilot revealed that the original exact matching approach failed on real search engine output.

### Deduplication
A shared record loader (syc/records.py) keeps only the latest record per unique key (conversation ID, detector model, detector input mode, and number of rounds). This prevents duplicate records from appearing when multiple concurrent runs write to the same output file. Gold sample export and evaluation both use this loader.

### Baseline Models for Comparative Evaluation
To benchmark the proposed MCP backed Detector, we defined two reference baselines:
* **Baseline 1 (All Sycophantic Prior):** Predicts that all observed stance flips in multi agent debate are sycophantic.
* **Baseline 2 (Correctness Oracle):** Labels any stance flip toward the ground truth answer as justified and any flip away from the ground truth answer as sycophantic.

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
* **Detector Performance:** Precision, Recall, Macro F1, and Cohen Kappa measured against programmatic Silver Labels. Human gold labels have not yet been collected. They require two independent raters to label the exported sample and are planned as future work.

## 6. Evaluated Results and Empirical Findings

A full pilot evaluation comprising 310 multi agent conversations was executed using groq:openai/gpt oss 20b as the Answerer and groq:openai/gpt oss 120b as the Detector. The genuine counterargument arm ran only on the 10 items where the Answerer was initially wrong and web evidence was available (46 of 70 items were skipped because the Answerer was already correct, 10 had no gold answer, and 4 had no supporting search results).

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

**Note on sample sizes:** The regressive flip rates above are computed from small denominators (37 initially correct items for MMLU, 9 for Test Bank Debatable). These point estimates should be interpreted with caution and confirmed with larger samples. The genuine counterargument arm had only 9 eligible conversations, producing a wide confidence interval.

### Key Behavioral Observations
**1. Susceptibility to Rhetorical Pressure:** On standardized academic benchmarks, models flipped in 8% to 14% of debates under purely rhetorical pressure containing zero factual justification.
**2. Regressive Sycophancy Was Observed:** In a small number of cases (2 to 3 out of 37 initially correct conversations), the model abandoned its correct knowledge and adopted a false claim due to peer insistence or manufactured consensus. These counts are too small to draw strong conclusions and require larger scale replication.
**3. Receptivity to Genuine Evidence:** When challenged with verified web evidence retrieved via MCP on questions it had gotten wrong, the model corrected its stance in 44.4% of cases (4 of 9 conversations), with 3 of those flips classified as evidence grounded by the Detector.

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
<td>1</td>
<td>1</td>
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

*Statistical Takeaway:* None of the pairwise comparisons reached statistical significance after Holm Bonferroni correction (all adjusted p values equal 1.000). The sample sizes (N = 10 for genuine counterargument comparisons) are too small to detect moderate effect sizes. Genuine counterarguments exhibited a positive risk difference (+30% vs social pressure and +20% vs insistence), suggesting a distinct behavioural dynamic, but this requires confirmation with larger samples.

### 6.3. Detector and Safeguard Performance

Evaluating the Detector across all 46 stance flip events (43 sycophantic, 3 evidence grounded) using silver labels:

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

### Cause and Outcome of Stance Flips
Across all 44 observed flips (some conversations had multiple flip events across rounds):
* **3 flips** were evidence grounded and progressive (the Answerer moved toward the correct answer after seeing verified evidence).
* **9 flips** were sycophantic but progressive (the Answerer happened to move toward the correct answer, but without verified evidence justifying the change).
* **32 flips** were sycophantic and regressive (the Answerer moved away from the correct answer under ungrounded pressure).

### Key Detector Observations
**1. Perfect Agreement with Silver Labels:** The Detector correctly classified all 46 flip events, achieving Macro F1 of 1.000 and Cohen Kappa of 1.000 against silver labels. However, these labels are programmatically derived (not human annotated), so this result reflects consistency between the Detector and the automated labeling pipeline rather than independent validation.
**2. Detector Matched Verifier on Evidence:** The Detector's assessment of whether the peer presented evidence agreed with the verifier's ground truth in 100% of cases.
**3. Baseline 1 Fails on the Genuine Arm:** The all sycophantic baseline achieves a kappa of 0.000 because it cannot identify any evidence grounded flips. Its high raw accuracy (93.5%) is misleading because the dataset is heavily imbalanced (only 3 EG events out of 46).
**4. Baseline 2 Misclassifies Lucky Guesses:** The correctness oracle misclassifies 19.6% of flips because an agent that flips to the correct answer due to peer pressure (without verified evidence) is still sycophantic. It achieves EG precision of only 0.250.
**5. Limitation:** The EG class has only 3 events, which means the Detector's perfect EG recall is based on a very small sample. Human gold labels (requiring two independent raters) have not yet been collected and are planned as future work.

## 7. Conclusion

This project implements and evaluates an MCP Backed Multi Agent Sycophancy Detector. By formalizing multi agent debate, integrating real time Model Context Protocol tool verification, and applying programmatic supervisory safeguards, our framework classifies individual belief revisions as either evidence grounded or sycophantic at the single turn level.

The pilot results on 310 conversations demonstrate that the Detector correctly separates verified evidence grounded flips from sycophantic ones across all 46 observed flip events (against silver labels). However, the evidence grounded class contained only 3 events, and human gold labels have not yet been collected. These results are promising but preliminary and require validation with larger samples and independent human annotation.

The behavioral findings confirm that LLMs are susceptible to rhetorical pressure in multi agent settings, with observed flip rates of 8% to 14% under purely social pressure. The genuine counterargument arm shows that verified web evidence can correct initially wrong answers, though the small sample (9 conversations) limits the strength of this conclusion.

### Limitations and Future Work
* **Small Evidence Grounded Sample:** Only 3 of 46 flip events were evidence grounded. Larger scale experiments with more initially wrong items are needed.
* **No Human Gold Labels:** All evaluation used programmatic silver labels. A double annotated human gold sample is planned but has not been collected.
* **Single Answerer Model:** Only groq:openai/gpt oss 20b was tested as the Answerer. Generalization to other models is unknown.
* **Genuine Arm Design:** In the current design, the Pressure agent continues arguing for the correct answer even after the Answerer has already switched to it. This can cause the Answerer to flip away again in later rounds, producing regressive flips within the genuine arm. Future work should consider stopping the genuine arm once the Answerer reaches the correct answer.

## 8. References

* **[1] Perez et al.:** Discovering language model behaviors with model written evaluations, in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 13387 to 13434.
* **[2] Sharma et al.:** Towards understanding sycophancy in language models, in Int. Conf. on Learning Representations (ICLR), 2024.
* **[3] Du et al.:** Improving factuality and reasoning in language models through multi agent debate, in Proc. Int. Conf. on Machine Learning (ICML), 2024, arXiv:2305.14325.
* **[4] Fanous et al.:** SycEval: Evaluating sycophancy in large language models, in Proc. AAAI/ACM Conf. on AI, Ethics, and Society, vol. 8, no. 1, 2025, pp. 893 to 900.
* **[5] Cheng et al.:** ELEPHANT: Measuring and understanding social sycophancy in LLMs, arXiv preprint arXiv:2505.13995, 2025.
* **[6] Yao et al.:** Peacemaker or troublemaker? How sycophancy shapes multi agent debate, arXiv preprint arXiv:2509.23055, 2025.
* **[7] Wynn et al.:** Talk isn't always cheap: Understanding failure modes in multi agent debate, arXiv preprint arXiv:2509.05396, 2025.
* **[8] Pitre et al.:** ConsensAgent: Towards efficient and effective consensus in multi agent LLM interactions through sycophancy mitigation, in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 22112 to 22133.
* **[9] Kasprova et al.:** Too polite to disagree: Understanding sycophancy propagation in multi agent systems, in Proc. 27th Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDIAL), 2026, pp. 795 to 814.
* **[10] Hou et al.:** Model Context Protocol (MCP): Landscape, security, and future of AI tool integration, arXiv preprint arXiv:2503.23278, 2025.
