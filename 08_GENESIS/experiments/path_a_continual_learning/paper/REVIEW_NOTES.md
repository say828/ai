# Paper Review Notes: Autopoietic Continual Learning v1

**Reviewer**: AI/ML Senior Expert
**Date**: 2026-01-04
**Document**: PAPER_DRAFT.md (v1)

---

## Executive Summary

The paper presents a novel approach to continual learning based on autopoietic theory. Overall, the work is well-structured with clear contributions and comprehensive experiments. However, several areas require improvement before submission.

**Overall Assessment**: Good foundation, needs refinement in clarity, equations, and interpretation.

---

## Section-by-Section Review

### 1. Abstract Review

**Checklist**:
- [x] 250단어 이내인가? - Currently ~280 words, **needs reduction**
- [x] 문제, 방법, 결과, 의의가 명확한가? - Yes, all present
- [x] 핵심 숫자 포함? - Yes (0.03%, 20x, p<0.001)
- [ ] Standalone하게 읽히는가? - Some jargon needs explanation

**Issues Identified**:
1. Too long (280 words vs 250 target)
2. Sentence "However, these methods typically rely on external loss functions and gradient-based optimization, which may not align with biological learning principles." is verbose
3. Missing key number: accuracy comparison (75.8% vs 99%)
4. "organizational closure" needs brief explanation

**Recommendations**:
- Condense to ~230 words
- Replace verbose phrases with concise alternatives
- Add one sentence clarifying the accuracy-forgetting trade-off upfront

---

### 2. Introduction Review

**Checklist**:
- [x] Motivation이 설득력 있는가? - Yes, catastrophic forgetting well-motivated
- [x] Gap이 명확한가? - Yes, biological plausibility gap identified
- [x] Our contribution이 4가지로 명확히 제시되었는가? - Yes
- [x] Key results가 구체적인가? - Yes

**Issues Identified**:
1. Section 1.2: Autopoiesis concept introduced too abruptly
2. The connection between autopoiesis and catastrophic forgetting could be stronger
3. RanPAC differentiation in Section 1.3 needs strengthening

**Recommendations**:
- Add a transition sentence before autopoiesis introduction
- Explicitly state why organizational closure prevents forgetting
- Add clearer contrast with RanPAC (learned vs random)

---

### 3. Related Work Review

**Checklist**:
- [x] 최소 25개 논문 인용? - Yes, 25 references listed
- [x] RanPAC, ESN, HebbCL 등 핵심 논문 포함? - Yes
- [x] 테이블로 비교가 명확한가? - Yes, Table in 2.2
- [ ] 우리의 novelty가 각 섹션에서 명시되는가? - **Needs improvement**

**Issues Identified**:
1. Section 2.2 comparison table is good but differentiation text is weak
2. Section 2.4 "first implementation" claim needs qualification
3. Missing some 2024-2025 papers (e.g., CODA-Prompt, L2P, DualPrompt)

**Recommendations**:
- Add "Critical Difference" subsection after Table in 2.2
- Qualify "first" claim with "to our knowledge"
- Add 2-3 more recent citations

---

### 4. Method Review

**Checklist**:
- [x] Algorithm이 명확한가? - Yes, pseudocode provided
- [ ] Coherence computation이 수식으로 명확한가? - **Needs formal equations**
- [x] Two-phase learning이 구분되어 설명되는가? - Yes
- [ ] Hyperparameter 선택 이유가 있는가? - Partially

**Issues Identified**:
1. Section 4.3: Coherence computation lacks formal mathematical definitions
2. The 4 components (predictability, stability, complexity, circularity) need equations
3. Weight values (0.2, 0.2, 0.15, 0.15, 0.3) don't sum to 1.0 consistently
4. Task_alignment in code not explained

**Critical Issues**:
- Line 247-249: Weights are 0.2+0.2+0.15+0.15+0.3 = 1.0, but task_alignment not defined in the text
- This is inconsistent with Section 3.2 (line 124) which has 4 components with different weights

**Recommendations**:
- Add formal equation block for each coherence component
- Clarify weight values and ensure consistency
- Remove or explain task_alignment

---

### 5. Experiments Review

**Checklist**:
- [x] Baseline 선택이 적절한가? - Yes
- [x] Statistical test가 올바른가? - Yes
- [x] Ablation이 모든 design choice를 cover하는가? - Yes
- [ ] Figure reference가 올바른가? - Figure 1 referenced but not defined

**Issues Identified**:
1. Ablation 1 interpretation: Random-Freeze achieving higher accuracy (90.13% vs 74.56%) is **surprising** and needs interpretation
2. Why does random initialization outperform learned?
3. Missing figure captions throughout
4. Per-task accuracy in Table 5 is good but needs interpretation

**Recommendations**:
- Add interpretation paragraph for Ablation 1's surprising finding
- Explain He initialization advantage
- Add comprehensive figure captions

---

### 6. Discussion Review

**Checklist**:
- [x] Limitation이 정직하게 언급되었는가? - Yes, 4 limitations listed
- [x] Accuracy-forgetting trade-off가 명확한가? - Yes
- [x] Future work가 구체적인가? - Yes, 6 directions
- [ ] Overclaim하지 않았는가? - "First implementation" needs qualification

**Issues Identified**:
1. Section 6.4 RanPAC comparison is fair
2. Limitations section is honest and comprehensive
3. Future work is realistic

**Recommendations**:
- Add "to our knowledge" qualification to first claim
- Strengthen conclusion section

---

### 7. Conclusion Review

**Issues Identified**:
1. Conclusion is functional but lacks impact
2. Missing strong closing statement
3. Could better emphasize paradigm shift

**Recommendations**:
- Rewrite conclusion with stronger narrative
- Add "three key contributions" structure
- End with memorable closing statement

---

### 8. Writing Quality

**Grammar/Style Issues**:
1. Inconsistent notation: W_in vs W_{in}
2. Some passive voice overuse
3. Minor: "~24%" should be consistent with "24%" or "approximately 24%"

**Notation Issues**:
- Line 38: W_in (code style)
- Line 159: W_{in} (LaTeX style)
- Recommendation: Use LaTeX style ($W_{in}$) consistently

---

## Priority Fixes (In Order)

### HIGH PRIORITY:

1. **Abstract condensation** - Reduce from 280 to 230 words
2. **Coherence equations** - Add formal LaTeX equations for all 4 components
3. **Ablation 1 interpretation** - Explain why random-freeze has higher accuracy
4. **Figure captions** - Add comprehensive captions for all figures

### MEDIUM PRIORITY:

5. **RanPAC differentiation** - Add "Critical Difference" subsection
6. **Conclusion strengthening** - Rewrite with stronger narrative
7. **Notation consistency** - Standardize to LaTeX notation

### LOW PRIORITY:

8. **Additional citations** - Add 2-3 more 2024-2025 papers
9. **Minor grammar fixes** - Passive voice, consistency
10. **Weight consistency** - Clarify coherence weight values

---

## Specific Text Changes Required

### Fix 1: Abstract
**Before**: "However, these methods typically rely on external loss functions and gradient-based optimization, which may not align with biological learning principles."

**After**: "However, these methods rely on external objectives and gradients, unlike biological learning."

### Fix 2: Section 2.2 Addition
Add after the comparison table:
```
**Critical Difference**: While RanPAC demonstrates that random fixed representations can work well, we hypothesize that *learning* task-relevant features before freezing provides additional benefits...
```

### Fix 3: Section 4.3 Equations
Replace code block with formal equations:
```
$$\text{Pred} = \frac{1}{1 + \text{Var}(\Delta h_t)}$$
$$\text{Stab} = \frac{1}{1 + \text{Var}(h_t)}$$
...
```

### Fix 4: Section 5.3.1 Interpretation
Add after Table 2:
```
**Surprising Finding**: Random-Freeze (RanPAC-style) achieves higher accuracy...
```

### Fix 5: Figure Captions
Add:
```
**Figure 1**: System architecture. (a) Overall: input passes through shared W_in...
```

### Fix 6: Conclusion Rewrite
Replace entire Section 7 with stronger version.

---

## Numerical Consistency Check

| Metric | Abstract | Results | Conclusion | Status |
|--------|----------|---------|------------|--------|
| Forgetting | 0.03% | 0.03% | 0.03% | OK |
| vs Fine-tuning | 20x | 20x | 20x | OK |
| p-value | p<0.001 | p=0.0004 | - | OK |
| Accuracy | ~75.8% | 75.75% | ~24% lower | OK |
| Ablation forgetting | - | 0.01% vs 0.04% | 4x | OK |

All numbers are consistent.

---

## Citation Check

**Core citations present**: [x] Kirkpatrick (EWC), [x] McDonnell (RanPAC), [x] Cossu (ESN), [x] Morawiecki (HebbCL), [x] Maturana & Varela

**Missing important citations** (recommended additions):
- Wang et al., 2024 - CL survey (already cited)
- Consider: L2P (Wang et al., 2022), DualPrompt (Wang et al., 2022)

---

## Final Verdict

**Ready for submission**: No
**Estimated effort to fix**: 2-3 hours
**After fixes**: Ready for internal review, then submission

