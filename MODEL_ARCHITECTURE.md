# Models and interpretation

The instantiated CLIP ViT-B/32 has a 512-dimensional projected global image embedding, a 512-dimensional text embedding, and 768-dimensional visual patch tokens. At its default 224×224 input, the 32-pixel patch grid has 49 tokens. Code extracts `vision_model(...).last_hidden_state[:, 1:]`; it does not hard-code token count. The CLS token is excluded. Backbone features are frozen, and each global/text embedding is L2 normalized. CLIP preprocessing is loaded with `use_fast=False` to retain historical behavior.

B1: torchvision pretrained ResNet50 with `ResNet50_Weights.DEFAULT` preprocessing and a nine-logit replacement head. The command supports either full training or explicitly frozen backbone training, and records that condition. B2: frozen CLIP image embedding and a linear 512→9 head, compatible with the preserved checkpoint. B3: image plus 28 numerical measurements. B4: image plus attribute-text embedding. B5: image, numerical measurements, and attribute text. B3–B5 concatenate inputs and use Linear→GELU→Dropout(0.1)→Linear with hidden dimension 256.

## Measurements

`features/aesthetic.py` defines ordered feature names and version `rgb256-v1`: RGB means, circular hue means, saturation mean/variance, value mean, warm/cool chromatic proportions, luminance mean/std, dark/bright proportions, percentile contrast, edge density, four edge quadrants, and eight saturation-weighted hue bins. Luminance standard deviation is the global contrast proxy. Edges use luminance gradients with a fixed threshold. No saliency measure is claimed. These are computed proxies, not aesthetic-quality judgments. Standardization uses training split means/std only; constant features receive a minimum standard deviation of 1e-6.

Descriptions use fixed brightness/contrast/warmth thresholds and contain no emotion names. They describe observed image properties without asserting causes, objects, intentions, or human responses.

## Attribute-Guided Fusion

Let `g∈R^512`, `a∈R^28`, `t∈R^512`. Each branch projects into `d=128`. With `u=concat(Pg(g),Pa(a),Pt(t))∈R^384`, compute `w=sigmoid(W2 GELU(W1 u))∈R^384`. The fused representation is `h=GELU(LayerNorm(Wf(u⊙w)))∈R^128`. Gating can be disabled for the concatenation control. A disabled numerical/text branch receives zeros; its projection biases remain trainable constants. Gating weights are not asserted to be faithful explanations.

## Emotion-Conditioned Spatial Attention

For each of nine classes, a linear projection of `h` supplies query `q_c∈R^128`. Shared projections transform all patch tokens into keys/values. `α_cp=softmax_p(q_c·k_p/√128)` and `r_c=Σ_p α_cp v_p`. The logit is `l_c=Wglobal_c h+b_c+vclass_c·r_c`. Thus regional evidence directly affects classification, and attention receives gradients from the classification loss. All maps sum to one over patches. The model optionally returns maps; a global-only ablation omits ECSA entirely.

Overlay geometry follows CLIP's actual resize and center crop. `explanations/geometry.py` maps spatial scores, masks, and patch boxes back onto the decoded original artwork. Pixels outside the encoder crop are marked as outside its spatial field of view. Geometry tests match the installed CLIP preprocessing for portrait, landscape, and square inputs. Inference preserves crop-space files and additionally writes original-artwork heatmaps, overlays, masks, and geometry metadata. Evidence perturbations use inverse-mapped regions on the original artwork and recompute every image-derived feature, preserving the deployed whole-image numerical branch. This measures the complete pipeline's sensitivity, not an isolated attention-module causal effect.

## Optional M2

Faithfulness-aware differentiable training is pending a stable trained M1 and a locked design. No M2 checkpoint or success is claimed. Implemented deletion/retention evaluation is a nondifferentiable image-space assessment, not a training loss. Future M2 must specify evidence area, regularity, gradient paths through masking and the frozen encoder, and the behavior of deterministic numerical/text extraction before claiming a faithfulness-aware objective.

Numerical-group masks are fixed buffers reconstructed from `attribute_indices` in the saved configuration. Their exclusion behavior is unit-tested. Text remains unchanged in these controls, so conclusions must refer specifically to the numerical branch.

## Explanation consistency and uncertainty

The structured explanation validates raw feature dimensions, finiteness, physical ranges, mutually exclusive pixel proportions, bounded dispersion, and brightness/mean consistency. Candidate regions must have unique nonnegative patch IDs, valid normalized attention mass, matching predicted class when supplied, and valid boxes within decoded image bounds when dimensions are known. Reports distinguish checks performed from unavailable geometry/class metadata; passing a bounds check is not a claim of semantic faithfulness.

Attribute wording comes directly from the measured features. When original-image boxes are available, the explanation identifies the highest-ranked candidate region by its geometric location without inventing objects or emotional causes. Stable class-order tie resolution agrees with the classifier's first-maximum convention. Predictive entropy and the top-two probability margin describe model uncertainty, explicitly not observer agreement or calibrated reliability. The stricter attribute checks passed all 14,000 training and 3,000 validation cached measurement vectors.
