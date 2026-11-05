# Neuronal Influence: Formalization and Practical Use

- **[Layer L]** Target internal layer where we focus the analysis. In this project we use:
  - `L = hidden before lm_head` (the vector feeding `lm_head.kernel`)
  - Optionally decoder GLU dense layers: `...GLU_0.Dense_0 (fc1)` and `...GLU_0.Dense_2 (fc2)`
- For an input example (image/text) `x`, denote the activation at `L` as `h(x) ∈ R^d`.

## EB selection (J, α)
- Excitation Backprop (EB) yields a relevance distribution on `L`.
- From EB we take:
  - `J ⊂ {1,…,d}`: indices of the most relevant neurons at `L` (top-k indices)
  - `α_j ≥ 0`: EB relevance values for those neurons (top-k values)
- These are not probabilities; they are relevance scores (positive evidence).

## Local objectives (F)
We define scalar objectives at layer `L` to steer the backward toward the EB-selected subspace:

- **Per-neuron objective**: `F_j(h) = h_j`
  - Meaning: we care about the activation of neuron `j` in `L`.
- **Aggregated objective**: `F_J(h) = ∑_{j∈J} α_j · h_j`
  - Meaning: we care about the EB-selected subspace, weighted by EB relevances.

These objectives are standard in gradient-based attribution: select a scalar quantity of interest and differentiate w.r.t. parameters or inputs.

## Influence per neuron
- Let `θ_j` denote the parameters associated with the output neuron `j` of the relevant layer (e.g., the `j`-th column of a dense/kernel). Then the per-example, per-neuron influence is

```
s_j(x) = || ∂F_j / ∂θ_j ||_2
```

- Implementation rule: “L2 norm across all tensor dimensions except the last (neuron/output dimension)”. For a 2D FC kernel `[in_dim, out_dim]`, this reduces to the L2 of column `j`.
- We use this exact reduction during training/offline export, so on-demand computations are comparable to the historical store.

## What it measures (interpretation)
- `s_j(x)` is a local sensitivity measure: how much the evidence for neuron `j` would change under small perturbations of its parameters, at the current model state, for example `x`.
- High `s_j(x)` ⇒ example `x` strongly “feeds” neuron `j` under the current parameterization.
- Caveat: this is not a causal intervention; it is a local (derivative-based) attribution consistent with EB’s relevances and with common gradient-based explanations.

## Practical pipelines
- **From history (recommended first):** compute/store `layer_influences[layer_name]` as per-neuron L2 norms during training or offline. Query top-k images per neuron using heaps without re-running backward.
- **On-demand (near real-time):** with EB indices `J` at `L`, define `F_J` or `F_j`, run VJP/grad only for those neurons and rank images by `s_j(x)`. Optionally prefilter candidates by activation `h_j(x)` (forward-only) for speed.

## References
- Zhang et al., “Top-Down Neural Attention by Excitation Backprop,” CVPR 2016.
- Bach et al., “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation,” PLoS ONE 2015.
- Simonyan et al., “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps,” ICLR 2014.
- Sundararajan et al., “Axiomatic Attribution for Deep Networks,” ICML 2017.
