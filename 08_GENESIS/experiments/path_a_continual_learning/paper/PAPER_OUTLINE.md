# Paper Outline: Autopoietic Continual Learning

**Target Venues**: NeurIPS, ICML, ICLR (ML Track), Frontiers in Computational Neuroscience (Bio-inspired)

---

## Section Overview

| Section | Pages | Words (est.) | Status |
|---------|-------|--------------|--------|
| Abstract | 0.3 | 250 | COMPLETE |
| 1. Introduction | 1.5 | 750 | COMPLETE |
| 2. Related Work | 2.0 | 1000 | COMPLETE |
| 3. Background (Autopoiesis) | 0.5 | 250 | COMPLETE |
| 4. Method | 2.5 | 1250 | COMPLETE |
| 5. Experiments | 3.0 | 1500 | COMPLETE |
| 6. Discussion | 1.5 | 750 | COMPLETE |
| 7. Conclusion | 0.5 | 250 | COMPLETE |
| References | 1.0 | - | COMPLETE (25 refs) |
| Appendix | 2.0 | 1000 | COMPLETE |
| **Total** | **~14 pages** | **~7000 words** | **COMPLETE** |

---

## Detailed Outline

### Abstract (250 words)
- [x] Problem: Catastrophic forgetting
- [x] Gap: Existing methods require external objectives
- [x] Approach: Autopoietic learning with coherence preservation
- [x] Key mechanism: Learn-then-freeze + Hebbian
- [x] Results: 0.03% forgetting (20x better)
- [x] Implications: Biological plausibility + minimal forgetting

### 1. Introduction (750 words)

#### 1.1 The Continual Learning Challenge (200 words)
- [x] Define catastrophic forgetting
- [x] Stability-plasticity dilemma
- [x] Real-world importance

#### 1.2 A Different Perspective: Autopoiesis (200 words)
- [x] Introduce Maturana & Varela
- [x] Organizational closure concept
- [x] Connection to neural networks

#### 1.3 Our Contribution (200 words)
- [x] Learn-then-freeze paradigm
- [x] Coherence-based acceptance
- [x] Hebbian plasticity
- [x] Empirical validation

#### 1.4 Key Results (150 words)
- [x] 20x lower forgetting statistic
- [x] Ablation summary
- [x] Trade-off acknowledgment

### 2. Related Work (1000 words)

#### 2.1 Continual Learning Approaches (300 words)
- [x] Regularization-based (EWC, SI, MAS)
- [x] Replay-based (GEM, A-GEM, iCaRL)
- [x] Architecture-based (Progressive NN, PackNet, DEN)

#### 2.2 Feature Freezing Methods (300 words)
- [x] RanPAC (random projections)
- [x] ESN (echo state networks)
- [x] RanDumb
- [x] Comparison table

#### 2.3 Biologically-Inspired Learning (250 words)
- [x] HebbCL
- [x] DHP
- [x] Gradient-free CL
- [x] Hebbian context gating

#### 2.4 Autopoiesis in ML (150 words)
- [x] Theoretical background
- [x] Gap: no CL implementation
- [x] Our contribution

### 3. Background: Autopoietic Systems (250 words)

#### 3.1 Autopoiesis Definition (100 words)
- [x] Maturana & Varela quote
- [x] Key properties

#### 3.2 Computational Formulation (100 words)
- [x] Organizational identity = W_in
- [x] Coherence metric formula
- [x] Structural drift condition

#### 3.3 Connection to CL (50 words)
- [x] Identity preservation = forgetting prevention
- [x] Intrinsic vs extrinsic constraint

### 4. Method (1250 words)

#### 4.1 Architecture (200 words)
- [x] Diagram
- [x] W_in, hidden, W_out components
- [x] Dimensions

#### 4.2 Learning Algorithm (600 words)
- [x] Phase 1: Task 0 (learn W_in + W_out)
- [x] Phase 2: Task t>0 (W_out only)
- [x] Pseudocode for both phases
- [x] Key insight explanation

#### 4.3 Coherence Computation (250 words)
- [x] Formula breakdown
- [x] Predictability, stability, complexity, circularity
- [x] Implementation code

#### 4.4 Key Design Choices (200 words)
- [x] Learn-then-freeze rationale
- [x] Threshold 0.95 rationale
- [x] Hebbian rationale
- [x] Task-specific heads rationale

### 5. Experiments (1500 words)

#### 5.1 Experimental Setup (250 words)
- [x] Dataset: Split-MNIST
- [x] Baselines: Fine-tuning, EWC, Replay
- [x] Hyperparameters table
- [x] Metrics definition

#### 5.2 Main Results (400 words)
- [x] Table 1: Main comparison
- [x] Statistical tests table
- [x] Key findings (4 points)
- [x] Trade-off discussion

#### 5.3 Ablation Studies (600 words)

##### 5.3.1 W_in Initialization (200 words)
- [x] Table 2
- [x] Analysis: random vs learned
- [x] Conclusion

##### 5.3.2 Coherence Threshold (200 words)
- [x] Table 3
- [x] Analysis: threshold effects
- [x] Conclusion

##### 5.3.3 Learning Rule (200 words)
- [x] Table 4
- [x] Analysis: Hebbian vs gradient
- [x] Conclusion

#### 5.4 Analysis (250 words)
- [x] Why does it work? (4 mechanisms)
- [x] Coherence-forgetting correlation
- [x] Per-task accuracy analysis

### 6. Discussion (750 words)

#### 6.1 Key Contributions (150 words)
- [x] First autopoietic CL
- [x] Near-zero forgetting
- [x] Biological plausibility
- [x] Novel criterion

#### 6.2 Limitations (150 words)
- [x] Accuracy trade-off
- [x] Single benchmark
- [x] Fixed architecture
- [x] Hebbian limitations

#### 6.3 Accuracy-Forgetting Trade-off (150 words)
- [x] Trade-off table
- [x] When to use our method
- [x] When to use gradient-based

#### 6.4 Comparison with RanPAC (150 words)
- [x] Detailed comparison table
- [x] Our novelty over RanPAC

#### 6.5 Future Work (150 words)
- [x] Larger benchmarks
- [x] Hybrid approaches
- [x] Online learning
- [x] Theory
- [x] Neuroscience validation

### 7. Conclusion (250 words)
- [x] Method summary
- [x] Results summary
- [x] Three key properties
- [x] Ablation validation
- [x] New research direction

### References (~25 citations)
- [x] Must-cite (7 papers)
- [x] Additional references (18 papers)

### Appendix

#### A. Extended Results
- [x] Per-trial results
- [x] Accuracy matrix
- [x] Coherence dynamics

#### B. Hyperparameter Sensitivity
- [x] Learning rate
- [x] Hidden dimension
- [x] Coherence threshold

#### C. Implementation Details
- [x] Pseudocode
- [x] Complexity analysis
- [x] Hardware/runtime

#### D. Statistical Analysis
- [x] t-test formula
- [x] Effect size formula
- [x] Full results table

---

## Figures (to include)

1. **Figure 1**: Architecture diagram (Input -> W_in -> Hidden -> W_out -> Output)
2. **Figure 2**: Learning curves (accuracy over tasks for each method)
3. **Figure 3**: Forgetting comparison bar chart
4. **Figure 4**: Ablation study 3x2 grid
5. **Figure 5**: Coherence vs forgetting scatter plot

**Existing figures in results/figures/**:
- accuracy_comparison.png
- forgetting_comparison.png
- forgetting_matrices.png
- learning_curves.png
- summary_dashboard.png

**Ablation figures in results/ablations/figures/**:
- ablation1_winit.png
- ablation2_coherence.png
- ablation3_learning_rule.png
- ablation_study_comprehensive.png
- ablation_statistical_significance.png

---

## Key Messages

### Main Claim
> Autopoietic continual learning achieves 20x lower forgetting than fine-tuning through organizational coherence preservation, providing a biologically-plausible alternative to gradient-based methods.

### Supporting Claims
1. Learn-then-freeze is superior to random-freeze for forgetting prevention (0.01% vs 0.04%)
2. Coherence threshold 0.95 is optimal for plasticity-stability balance
3. Hebbian learning trades ~15% accuracy for 10x lower forgetting

### Honest Limitations
- 24% accuracy trade-off vs gradient methods
- Only tested on Split-MNIST
- Slower convergence than SGD/Adam

---

## Reviewer Concerns to Address

### Likely Questions

1. **"How is this different from RanPAC?"**
   - We LEARN W_in (task-relevant), they use RANDOM
   - We use coherence acceptance, they don't
   - We achieve 4x lower forgetting (0.01% vs 0.04%)

2. **"Why such low accuracy?"**
   - Explicit trade-off for biological plausibility
   - Hebbian learning is less efficient than backprop
   - Future work: hybrid approaches

3. **"Only Split-MNIST?"**
   - Acknowledge limitation
   - Focus on proof-of-concept + rigorous ablations
   - Future work includes larger benchmarks

4. **"Is this biologically plausible?"**
   - More plausible than backprop (local updates)
   - Coherence = homeostasis in biology
   - But still simplified model

---

## Submission Timeline

| Task | Deadline | Status |
|------|----------|--------|
| Draft complete | 2026-01-04 | DONE |
| Internal review | 2026-01-07 | TODO |
| Revision | 2026-01-10 | TODO |
| Figure polish | 2026-01-12 | TODO |
| Final proofread | 2026-01-14 | TODO |
| Submit | NeurIPS/ICML deadline | TODO |

---

*Outline generated: 2026-01-04*
