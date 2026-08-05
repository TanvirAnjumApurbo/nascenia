# Nasenica AI Hackathon: Official Rulebook

**Task:** Bengali Medical Dialogue Generation
**Platform:** Kaggle Competitions
**Model constraint:** ≤ 3B parameters

## 1. Overview

Participants are required to develop a system that generates a doctor's response in Bengali based on a patient's prompt. Submissions will be evaluated for both semantic and textual similarity to the reference response, as well as medical diagnostic accuracy. Access to quality healthcare guidance is a challenge for millions of Bengali speakers, and most existing medical NLP tools are built for English. This competition challenges you to build an AI system that can act as a doctor — reading a patient's prompt in Bengali and generating a clinically sound, well-communicated response, the way a real physician would.

The competition runs in two phases:

- **Phase 1: Leaderboard Phase:** Automated scoring against a held-out test set via Kaggle's public/private leaderboard split.
- **Phase 2: Verification & Judging Phase:** The top 10 teams from the Phase 1 leaderboard submit their model + inference script for parameter verification and LLM-as-judge evaluation.

Final ranking is a weighted combination of both phases: Phase 1 leaderboard score contributes 80%, and Phase 2 LLM-as-judge score contributes 20%. Only entries that pass parameter verification in Phase 2 are eligible for final ranking and prizes.

## 2. Dataset

- **Language:** Bengali
- **Input:** A patient's prompt/query to a doctor
- **Output (target):** The expected doctor's response
- **Splits:** Train (released to participants) / Public test (scored live on leaderboard) / Private test (revealed after competition close)
- Participants may only use the provided training data plus any external data for fine-tuning. External data usage must be disclosed in the submission notes.
- Dataset is provided for the sole purpose of this competition; redistribution outside the competition is prohibited.

## 3. Model Constraints

- Any base LLM may be used as long as it has no more than 3,000,000,000 (3B) parameters at inference time.
- Fine-tuning, LoRA/adapters, quantization, distillation, and prompt-engineering are all permitted, provided the resulting deployed inference model stays under the 3B parameter cap.
- Ensembling multiple models is permitted, provided that the combined parameter count of all models used at inference stays under 3B.
- Parameter count is verified in Phase 2 by inspecting the submitted model weights and inference script. Misrepresenting parameter count is grounds for disqualification (see Section 8).

## 4. Evaluation Metric (Phase 1: Leaderboard)

A custom composite score is used:

| Component      | Weight |
| :------------- | :----- |
| BERTScore      | 50%    |
| Token-level F1 | 30%    |
| ROUGE-L F1     | 20%    |

Phase 1 Score = 0.5 × BERTScore + 0.3 × Token F1 + 0.2 × ROUGE-L F1

- Scored per-example against the reference doctor response, then averaged across the test set.
- The public leaderboard reflects score on the public portion of the test set and updates in real time as teams submit.
- The private leaderboard reflects score on the held-out private portion and is revealed immediately after the competition submission deadline. Private leaderboard rank determines who advances to Phase 2: top 10 teams/individuals advance.
- The private leaderboard score is the Phase 1 Score carried forward into the final weighted ranking (Section 5.4).

## 5. Phase 2: Verification & LLM-as-Judge

Only the top 10 entries from the Phase 1 private leaderboard advance.

### 5.1 Required Submissions

Each qualifying team must submit, by August 25th, 12 PM, BD Time Zone:

1. Full inference script (reproducible, documented, runnable end-to-end)
2. Model weights/checkpoint (or a script to download them from a reproducible source)
3. A short write-up describing the approach, base model used, fine-tuning method, and any external data/tools used
4. Environment/dependency file.

### 5.2 Parameter Verification

- Organizers will load the submitted model and confirm total parameter count ≤ 3B.
- Any entry exceeding the limit, or whose inference script does not reproduce the leaderboard-submitted outputs within a reasonable tolerance, will be disqualified and replaced by the next-highest-ranked eligible entry.

### 5.3 LLM-as-Judge Evaluation

- A held-out judging set (separate from the private test set) is used.
- An LLM judge scores each of the top 10 models' outputs on:
  - **Medical diagnostic accuracy:** is the response clinically appropriate/correct?
  - **Response quality/appropriateness:** tone, completeness, clarity as a doctor's response
- To reduce bias, judging will be blind (team identities hidden from the judge) and outputs will be presented in randomized order.
- Organizers reserve the right to include a human review step to spot-check LLM judge outputs for reliability.
- Judge scores are normalized to a 0-1 scale to produce the Phase 2 Score, used in the final weighted ranking below.

### 5.4 Final Weighted Ranking

For each of the top 10 verified entries:
Final Score = 0.8 × Phase 1 Score + 0.2 × Phase 2 Score
(Phase 1 and Phase 2 scores are normalized to the same scale before combining.) Final rank and prize placement are determined by this weighted Final Score among the top 10 verified entries. Any entry that fails parameter verification (Section 5.2) is excluded from final ranking regardless of its Phase 1 or Phase 2 score.

## 6. Eligibility & Teams

- Open to anyone.
- Team size: up to 4 members per team. Teams must be finalized by August 2nd; no changes after this date.
- Each individual may be part of only one team.
- Organizers, sponsors and their immediate family members are not eligible to compete.
- Participants must have a valid Kaggle account and agree to Kaggle's competition rules in addition to this rulebook.

## 7. Timeline

| Milestone                                                        | Date                             |
| :--------------------------------------------------------------- | :------------------------------- |
| Competition opens / data released                                | August 4th                       |
| Team registration deadline                                       | August 2nd                       |
| Phase 1 submission deadline (leaderboard closes)                 | August 24th, 00:00, BD Time Zone |
| Private leaderboard revealed                                     | Immediately after contest end    |
| Top 10 announced / Phase 2 submission opens                      | Immediately after contest end    |
| Phase 2 (model + inference script) submission deadline           | August 25th, 00:00, BD Time Zone |
| LLM-as-judge results, final weighted ranking & winners announced | August 26th                      |

All deadlines are in GMT+6 unless otherwise stated.

## 8. Submission Rules & Fair Play

- Submission limit: 5 during Phase 1 via Kaggle.
- Final leaderboard rank is based on the last submission / the submission the participant selects as final before the deadline.
- Participants must not:
  - Attempt to access, reverse-engineer, or manually label the private test set
  - Use the test set inputs for training or fine-tuning
  - Submit outputs generated by manual/human labeling instead of their model
  - Use models exceeding the 3B parameter cap, or misrepresent parameter count
  - Share private leaderboard test data or Phase 2 submissions with other teams
- Organizers may disqualify any entry found to violate these rules, at any stage, including after prizes are announced.
- Organizers reserve the right to request original training logs, code, or additional proof of compliance from any participant.

## 9. Prizes

- **1st place:** 30,000 BDT
- **2nd place:** 10,000 BDT
- **3rd place:** 5,000 BDT

Prizes are awarded according to the final weighted Final Score (Section 5.4). Winners must complete the verification steps outlined in Section 5 and may be required to provide identity or payment verification as part of Kaggle's standard prize-competition process. Unless otherwise specified, winners are responsible for all tax and payment arrangements.

## 10. Intellectual Property & Licensing

- Participants retain ownership of their model/code, but by submitting, grant organizers a license to use, evaluate, and for winning entries showcase the model and write-up for the purposes of the competition, with attribution.
- The dataset remains the property of Nascenia LTD and is licensed for competition use only under CC BY-NC 4.0.
- Winning teams may be asked to open-source their inference code as a condition of receiving the prize.

## 11. Ethical & Medical Disclaimer

- This competition is for research and educational purposes. Models produced are not validated for real clinical use and must not be deployed as an actual medical advice tool without proper regulatory review.
- Participants should be mindful that outputs deal with medical/health content; any submission containing harmful, unsafe, or clearly unethical generated medical advice may be flagged during LLM-as-judge review and penalized.

## 12. Organizer Rights

- Organizers reserve the right to update these rules, extend deadlines or adjust the evaluation methodology if necessary, with notice posted on the competition page.
- In case of a tie for 1st–3rd place in the final weighted Final Score, there will be a rerun of Phase 2 for tiebreaker between tied participants.
- Organizers' decisions on disqualification, verification, and final ranking are final.

## 13. Contact & Support

- **Questions:** wasiahmad@nascenia.com
- Please report data issues or leaderboard bugs to wasiahmad@nascenia.com

By participating in this competition, you agree to abide by this rulebook as well as Kaggle's Competition Rules / Terms of Use.


# From Kaggle


### The Challenge

You'll be given a dataset of Bengali patient–doctor dialogue pairs: a patient's prompt describing their symptoms or concerns, and the expected doctor's response. Your task is to fine-tune a language model that generates doctor-quality responses for unseen patient prompts — responses that are not just fluent and relevant, but medically accurate.

**The catch:** your model must be lightweight. We're capping base LLM size at **3 billion parameters**, so this competition is about efficient fine-tuning and clever adaptation, not simply scaling up to the largest model available.

### How You'll Be Judged

Scoring happens in two phases:

1. **Phase 1 (Leaderboard):** Your submissions are automatically scored on Kaggle using a composite metric — 50% BERTScore, 30% Token-level F1, and 20% ROUGE-L F1 — measuring how closely your generated response matches the reference doctor's response. Public and private leaderboards track your progress throughout the competition.
2. **Phase 2 (Verification & LLM-as-Judge):** The top 10 leaderboard entries submit their model and inference script. After we verify the parameter limit, an LLM judge evaluates the outputs for medical diagnostic accuracy and response quality. Final rankings combine both phases — 80% from your Phase 1 leaderboard score, 20% from the Phase 2 LLM-judge score.

### Why It Matters

This isn't just a leaderboard exercise — it's a step toward making AI-assisted healthcare guidance accessible in Bengali, one of the world's most widely spoken and most underserved languages in NLP. Strong solutions here could inform real tools that help patients get clearer, faster guidance before they ever see a doctor.

---

## Description

Bengali is spoken by over 230 million people, yet it remains one of the most underserved languages in medical NLP. Most AI health-assistant tools are built and tested in English, leaving a huge population without access to AI-supported medical guidance in their native language.

This competition asks you to help close that gap. Using a dataset of real Bengali patient prompts paired with expected doctor responses, your goal is to fine-tune a language model that can read a patient's description of their symptoms and generate a response with the accuracy, clarity, and tone of an actual doctor.

To keep the playing field focused on skill rather than scale, base models are capped at **3 billion parameters** — so success here comes down to smart fine-tuning and adaptation, not simply picking the biggest model available.

Submissions are scored on a composite metric of semantic similarity (BERTScore, Token F1, ROUGE-L), and the top entries advance to a second round where an LLM judge evaluates medical diagnostic accuracy and response quality directly. Final placements and prizes are decided by a weighted combination of both rounds.

Whether you're an NLP researcher, a healthcare AI enthusiast, or just looking to sharpen your fine-tuning skills, this is a chance to build something that could genuinely help people get clearer answers to their health questions — in their own language.

---

## Evaluation

### Overview

Submissions are evaluated on how closely the generated doctor response matches the reference (ground-truth) doctor response, using a custom composite metric that blends semantic similarity with lexical overlap.

### Metric

For each patient prompt, your model generates a predicted response $\hat{y}$, which is compared against the reference response $y$. Three sub-metrics are computed per example:

1. **BERTScore F1** — measures semantic similarity using contextual embeddings.
   $$ \text{BERTScore}_{F1} = \frac{2 \cdot P_{BERT} \cdot R*{BERT}}{P*{BERT} + R*{BERT}} $$
   where $P*{BERT}$ and $R_{BERT}$ are the BERTScore precision and recall between the token embeddings of $\hat{y}$ and $y$.

2. **Token-level F1** — measures word-overlap between prediction and reference, treating each as a bag of tokens.
   $$ \text{Token F1} = \frac{2 \cdot P*{tok} \cdot R*{tok}}{P*{tok} + R*{tok}} $$
where:
$$ P*{tok} = \frac{|\text{tokens}(\hat{y}) \cap \text{tokens}(y)|}{|\text{tokens}(\hat{y})|}, \quad R*{tok} = \frac{|\text{tokens}(\hat{y}) \cap \text{tokens}(y)|}{|\text{tokens}(y)|} $$

3. **ROUGE-L F1** — measures the longest common subsequence (LCS) overlap between prediction and reference.
   $$ \text{ROUGE-L}_{F1} = \frac{(1+\beta^2) \cdot R_{lcs} \cdot P*{lcs}}{R*{lcs} + \beta^2 \cdot P*{lcs}} $$
where:
$$ R*{lcs} = \frac{\text{LCS}(\hat{y}, y)}{|y|}, \quad P\_{lcs} = \frac{\text{LCS}(\hat{y}, y)}{|\hat{y}|} $$
and $\beta$ is set such that precision and recall are weighted equally (standard ROUGE-L convention, $\beta = 1$).

### Final Composite Score (Phase 1)

The three sub-metrics are combined into a single per-example score:
$$ \text{Score} = 0.5 \cdot \text{BERTScore}_{F1} + 0.3 \cdot \text{Token F1} + 0.2 \cdot \text{ROUGE-L}_{F1} $$

Your Phase 1 leaderboard score is the mean of this composite score across all examples in the test set:
$$ \text{Phase 1 Score} = \frac{1}{N} \sum\_{i=1}^{N} \text{Score}\_i $$
where $N$ is the number of examples in the (public or private) test split.

### Phase 2: LLM-as-Judge Score

For the top 10 leaderboard entries, an LLM judge scores each model's outputs on medical diagnostic accuracy and response quality/appropriateness, on a normalized **0–100** scale. The mean judge score across the held-out judging set forms the Phase 2 Score.

### Final Ranking

Final placement among the top 10 verified entries is determined by a weighted combination of both phases:
$$ \text{Final Score} = 0.8 \cdot \text{Phase 1 Score} + 0.2 \cdot \text{Phase 2 Score} $$
(Phase 1 and Phase 2 scores are normalized to the same 0–100 scale before combining.) Entries that fail parameter verification are excluded from final ranking regardless of score.

### Submission Format

For each `id` in the test set, submit your model's predicted doctor response. The file should contain a header and have the following format:

```csv
id,doctor_response
1,আপনার লক্ষণগুলো শুনে মনে হচ্ছে...
2,এটি সাধারণত...
...
```


---

## Dataset Description

### Overview

This dataset contains Bengali patient–doctor dialogue pairs. Each example consists of an input — a patient's prompt describing their symptoms, concerns, or a medical question, written as a patient would address a doctor — and, for training data, an output — the expected doctor's response, including relevant medical guidance.

Your task is to build a model that, given a new patient prompt, generates a response that matches the quality, accuracy, and tone of an actual doctor's reply.

### Files

- **train.csv** — 108,954 labeled examples. Use this to fine-tune your model.
- **test.csv** — 1,000 unlabeled patient prompts for which you must generate doctor responses. This is split into a public and private portion; your leaderboard score during the competition reflects performance on the public portion only.

### Columns

| Column     | Description                                              |
| ---------- | -------------------------------------------------------- |
| **id**     | Unique identifier for each example                       |
| **input**  | The patient's prompt to the doctor, in Bengali           |
| **output** | _(train only)_ The reference doctor response, in Bengali |

### Submission Format

For each `id` in `test.csv`, submit the generated doctor response. Your `submission.csv` should have exactly two columns:

```csv
id,output
34654,আপনার লাইপেজ লেভেল বৃদ্ধির কারণ...
3116,আপনার এই লক্ষণগুলোর জন্য...
...

```

> **A Note on the Data:**
> This dataset deals with real medical dialogue content. It is provided strictly for research and competition purposes under a CC BY-NC 4.0 license, and is not intended to be used as an actual clinical decision-making resource. See the Rules tab for details on data access, use, and redistribution restrictions.




