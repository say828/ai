# Pre-Submission Checklist: Autopoietic Continual Learning

**Paper Title**: Autopoietic Continual Learning: Preventing Catastrophic Forgetting Through Organizational Coherence

**Target Venues**: NeurIPS, ICML, ICLR

---

## 1. Technical Correctness

### 1.1 Experimental Rigor
- [x] Multiple trials (N=3 main, N=5 ablations)
- [x] Standard deviations reported
- [x] Statistical significance tests (t-tests)
- [x] Effect sizes (Cohen's d) reported
- [x] Random seeds fixed for reproducibility

### 1.2 Fair Comparisons
- [x] Same hyperparameter tuning budget for all methods
- [x] Same training epochs per task
- [x] Same evaluation protocol
- [x] FLOPs counted for computational fairness
- [ ] Consider adding more baselines (SI, MAS, A-GEM)?

### 1.3 Metric Validity
- [x] Forgetting metric defined clearly (standard definition)
- [x] Average accuracy computed correctly
- [x] Per-task breakdown provided in appendix
- [x] Accuracy matrices shown

---

## 2. Claims and Limitations

### 2.1 Main Claims
| Claim | Evidence | Status |
|-------|----------|--------|
| 20x lower forgetting vs fine-tuning | 0.03% vs 0.60%, p<0.001 | Supported |
| Comparable to EWC | 0.03% vs 0.06%, p=0.065 | Supported (trending) |
| Learned-freeze > random-freeze | 0.01% vs 0.04% | Supported |
| Coherence criterion validated | 74.6% vs 65.5% (no check) | Supported |
| Hebbian trades accuracy for forgetting | 10x lower forgetting | Supported |

### 2.2 Limitations Acknowledged
- [x] Accuracy trade-off (~24% lower)
- [x] Single benchmark (Split-MNIST only)
- [x] Fixed architecture (no expansion)
- [x] Hebbian convergence slower
- [x] Not tested on large-scale datasets

### 2.3 Claims NOT Made
- [ ] Do NOT claim state-of-the-art accuracy
- [ ] Do NOT claim biological realism (only plausibility)
- [ ] Do NOT claim scalability to ImageNet (not tested)

---

## 3. Related Work Coverage

### 3.1 Must-Cite Papers (CRITICAL)
| Paper | Cited | Differentiated |
|-------|-------|----------------|
| RanPAC [McDonnell 2023] | [x] | [x] - We learn, they random |
| ESN for CL [Cossu 2021] | [x] | [x] - We learn, they random |
| HebbCL [Morawiecki 2023] | [x] | [x] - We add coherence criterion |
| EWC [Kirkpatrick 2017] | [x] | [x] - Baseline |
| Progressive NN [Rusu 2016] | [x] | [x] - Architecture vs fixed |
| GEM [Lopez-Paz 2017] | [x] | [x] - Replay-free |
| Three Types of IL [van de Ven 2022] | [x] | [x] - Scenario definition |

### 3.2 Additional Important Papers
- [x] SI [Zenke 2017]
- [x] LwF [Li & Hoiem 2016]
- [x] iCaRL [Rebuffi 2017]
- [x] PackNet [Mallya 2018]
- [x] DEN [Yoon 2018]
- [x] A-GEM [Chaudhry 2019]
- [x] MAS [Aljundi 2018]
- [x] DHP [Miconi 2018]
- [x] Maturana & Varela 1980 (autopoiesis)

### 3.3 Potential Missing Citations
- [ ] Recent 2024-2025 continual learning papers?
- [ ] Forward-Forward algorithm [Hinton 2022]
- [ ] CODA-Prompt, L2P for Vision Transformers?

---

## 4. Presentation Quality

### 4.1 Writing
- [ ] Abstract: Clear problem/method/result
- [ ] Introduction: Motivation established
- [ ] Method: Reproducible from text
- [ ] Results: Tables/figures clear
- [ ] Discussion: Honest about limitations

### 4.2 Figures
| Figure | Location | Quality |
|--------|----------|---------|
| Architecture diagram | Section 4.1 | Needed |
| Learning curves | results/figures/ | Exists |
| Forgetting comparison | results/figures/ | Exists |
| Ablation 3x2 grid | results/ablations/figures/ | Exists |
| Coherence-forgetting scatter | Needed | TODO |

### 4.3 Tables
- [x] Table 1: Main results comparison
- [x] Table 2: W_in initialization ablation
- [x] Table 3: Coherence threshold ablation
- [x] Table 4: Learning rule ablation
- [x] Table 5: Per-task accuracy breakdown

### 4.4 Formatting
- [ ] Page limit compliance (8-10 pages + appendix)
- [ ] Font size correct
- [ ] Margins correct
- [ ] References formatted consistently
- [ ] Anonymous submission (if double-blind)

---

## 5. Ethical Considerations

### 5.1 Dataset
- [x] MNIST is public domain
- [x] No privacy concerns
- [x] Standard benchmark

### 5.2 Broader Impact
- [x] Positive: Reduces catastrophic forgetting
- [x] Positive: More efficient (no replay storage)
- [x] Positive: Biologically plausible
- [ ] Consider: Applications in sensitive domains?

### 5.3 Reproducibility
- [x] Code will be released
- [x] Hyperparameters specified
- [x] Random seeds provided
- [x] Hardware requirements minimal (CPU only)

---

## 6. Reviewer Anticipation

### 6.1 Likely Positive Points
- Novel theoretical framing (autopoiesis)
- Comprehensive ablations
- Honest about trade-offs
- Biologically plausible approach

### 6.2 Likely Concerns

| Concern | Our Response |
|---------|--------------|
| "Just feature freezing, not novel" | We LEARN then freeze (not random); coherence criterion is novel |
| "Low accuracy" | Explicit trade-off; appropriate for stability-critical applications |
| "Only MNIST" | Proof-of-concept with rigorous ablations; future work includes larger benchmarks |
| "How is coherence different from early stopping?" | Coherence is multi-dimensional (predictability, stability, complexity, circularity), not just accuracy |
| "Why not compare with more methods?" | Focus on key representative methods; can add more in revision |

### 6.3 Questions to Prepare For

1. **"What happens with more tasks (10, 20, 50)?"**
   - We expect consistent near-zero forgetting due to frozen W_in
   - Not tested; acknowledge as future work

2. **"What about class-incremental learning (no task IDs)?"**
   - Currently task-incremental only
   - Class-IL is harder; potential extension

3. **"How does hidden dimension affect forgetting?"**
   - Ablation in Appendix B.2 shows minimal effect
   - 256 is sufficient for Split-MNIST

4. **"Why Hebbian specifically?"**
   - Biological plausibility
   - Local updates = no backprop
   - Historical significance (Hebb 1949)

---

## 7. Final Checks

### 7.1 Before Submission
- [ ] Spell check entire document
- [ ] Check all figure references
- [ ] Check all table numbers
- [ ] Verify citation format
- [ ] Remove any identifying information (if anonymous)
- [ ] Check supplementary material compiles

### 7.2 Code Release
- [ ] Clean up code
- [ ] Add requirements.txt
- [ ] Write README.md
- [ ] Include example commands
- [ ] Verify reproducibility on fresh install

### 7.3 Supplementary Material
- [ ] Extended ablations
- [ ] Additional visualizations
- [ ] Full hyperparameter tables
- [ ] Per-trial raw data

---

## 8. Submission Status

| Item | Status | Notes |
|------|--------|-------|
| PAPER_DRAFT.md | COMPLETE | Full draft |
| PAPER_OUTLINE.md | COMPLETE | Section breakdown |
| Figures | PARTIAL | Need architecture diagram |
| Code | EXISTS | In repository |
| Supplementary | TODO | Need to compile |

### Next Steps

1. [ ] Create architecture diagram (Figure 1)
2. [ ] Polish writing (human review)
3. [ ] Add any missing citations
4. [ ] Convert to LaTeX (for venue)
5. [ ] Final proofreading
6. [ ] Submit!

---

## 9. Confidence Assessment

### Overall Submission Readiness: **85%**

| Aspect | Confidence | Notes |
|--------|------------|-------|
| Technical correctness | 95% | Rigorous experiments |
| Novelty | 80% | Clear differentiation from RanPAC needed |
| Presentation | 75% | Needs polishing |
| Impact | 70% | Niche but interesting |
| Reproducibility | 95% | Code available |

### Recommended Actions

1. **High Priority**:
   - [ ] Strengthen differentiation from RanPAC in introduction
   - [ ] Add architecture figure
   - [ ] Polish abstract

2. **Medium Priority**:
   - [ ] Add more recent citations (2024-2025)
   - [ ] Consider additional baseline (SI or MAS)
   - [ ] Expand future work section

3. **Low Priority**:
   - [ ] Additional hyperparameter ablations
   - [ ] Longer training experiments
   - [ ] Different random seeds

---

*Checklist generated: 2026-01-04*
*Last updated: 2026-01-04*
