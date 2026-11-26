# ExplainVision project plan

This document is the working plan for ExplainVision. [README.md](README.md) is the public summary. It receives a measured number only after the evidence rule below is met. This commit contains the plan and the pipeline figure. The code, the demo, and the evaluation harness are still to be written. [Milestones](#milestones) is the checklist.

The pipeline the project implements is the one drawn in [docs/pipeline.svg](docs/pipeline.svg):

input image → preprocess → CNN classifier → prediction (top-k); from the CNN, target-layer activations and gradients → explanation method → heatmap → overlay on the image.

## Goals

- Show, on a single image, why a convolutional classifier assigned a class, with more than one explanation method side by side.
- Load two kinds of model: the CIFAR-10 CNN from [VisionClassify](https://github.com/cendysaputra/VisionClassify), and ImageNet-pretrained torchvision CNNs (ResNet-50 and an EfficientNet) for higher-resolution photographs.
- Evaluate the explanations, not only the classifier: deletion AUC, insertion AUC, a parameter-randomization sanity check, and runtime per method.
- Show a model-failure gallery, so heatmaps appear on images the classifier got wrong.
- Keep every number in the README traceable to a script in `eval/`, with the log, the metrics JSON, and the model and config versions saved beside it.
- Leave the code readable enough to sit next to VisionClassify, AudioScope, VecScope, and DetectIt.

## Non-goals

- Training a new classifier in order to raise accuracy. The project explains models that already exist.
- A production labeling service, an account system, or a hosted deployment as part of the first version.
- A claim that a heatmap makes a model safe, fair, or fit for a medical or legal decision. The maps are a learning tool.
- Explaining architectures that do not have spatial convolutional feature maps. A Vision Transformer is a different project. If VisionClassify's default checkpoint has moved on to one, this repository still loads the CNN, pinned by commit.
- Object-detection explanations, which belong with [DetectIt](https://github.com/cendysaputra/DetectIt), and audio explanations, which belong with [AudioScope](https://github.com/cendysaputra/AudioScope).
- Declaring a winning explanation method before the metrics exist.
- Reimplementing every method from scratch when pytorch-grad-cam and Captum already provide them. Wrappers are the implementation. A from-scratch Grad-CAM is in scope only as a check against the library on the same tensor, and only one of the two is used in the app.
- Redistributing CIFAR-10, ImageNet, or pretrained weights in git history.
- Copying a leaderboard or model-card accuracy into the README and presenting it as a run from this repository.

## Background

These are the ideas the explainers depend on. The implementation will follow the definitions here so the evaluation scripts and the Streamlit app compute the same maps.

### Feature maps

A convolutional layer emits a stack of spatial grids. Each grid is one channel, often called a feature map. A cell in that grid is a location in the image, downsampled by the strides and pooling up to that layer, and the value is how strongly that channel fired there. Early layers respond to edges and colors. Later layers respond to patterns that are closer to the class decision, on a coarser grid. The heatmap methods in the Grad-CAM family are combinations of these maps. They cannot be finer than the grid of the layer they read, until an upsampling step stretches them back to the image.

### Gradients

A gradient here is the partial derivative of a scalar score with respect to a tensor the network computed. The score is the logit of one class, the value before the softmax. Softmax gradients shrink once the model is confident, so they are a poor signal for an explanation, and this project does not use them as the quantity being differentiated.

Two gradients matter:

- The gradient of the class logit with respect to a convolutional feature map. Grad-CAM and Grad-CAM++ use this. It answers how a small change in that channel, at that location, would move the class score.
- The gradient of the class logit with respect to the input pixels. Saliency maps and Integrated Gradients use this. It answers how a small change in a pixel would move the class score.

Backpropagation computes both. A method that never calls backward is a perturbation method, described below.

### Target layer

The target layer is the convolutional layer an activation-based method reads. For Grad-CAM this is usually the last convolutional layer: the channels are semantic, and the grid still has a spatial layout. An earlier layer is finer and less tied to the class. A fully connected classification layer has no spatial grid, and a class activation map cannot be read from it.

Each model config names its default target layer. A test checks that the name resolves to a module and that the module's output is a 4-dimensional tensor (batch, channels, height, width). The app may override the layer, and the chosen name is stored with every run and shown in the demo. Integrated Gradients and saliency do not use a target layer. The UI says so instead of pretending they read one.

### Class activation maps

A class activation map is a single spatial heatmap for one class. High values mark regions that support that class under the method's definition.

Zhou et al. (2016) built the original class activation map for networks that end in global average pooling and a linear classifier. The linear weight of each channel is the channel's importance, and the map is the weighted sum of the last convolutional feature maps. That construction needs the global-average-pooling architecture.

Grad-CAM (Selvaraju et al., 2017) keeps the weighted sum and replaces the linear weight with a weight computed from gradients. The same idea then applies to a CNN that does not end in global average pooling, which covers the VisionClassify CNN and the torchvision models in this project. The map is still only as detailed as the target layer, and the ReLU inside Grad-CAM keeps positive support for the class and drops locations that count against it.

### How Grad-CAM works

The steps below are the procedure in Selvaraju et al. (2017). Grad-CAM++ and Score-CAM change the way the channel weights are computed. They still end at a heatmap on the target-layer grid.

1. **Forward.** Run the preprocessed image through the CNN. Keep the activations of the target layer. Call the feature map of channel k by the name A^k. There are K channels, each a spatial grid.
2. **Score.** Take y^c, the logit of the class c being explained. This is the value before the softmax.
3. **Backward.** Back-propagate y^c to the target layer. The result is the gradient of the logit with respect to every spatial cell of every channel.
4. **Channel weights.** Average each channel's gradient over its spatial cells. That average is alpha_k^c, the importance of channel k for class c. In the paper, Z is the number of cells:

   ```text
   alpha_k^c = (1 / Z) * sum over i, j of  (d y^c / d A^k_ij)
   ```

5. **Weighted sum, then ReLU.** Combine the feature maps and drop negative locations. The ReLU is applied once, to that weighted sum. Locations where L^c is positive are the ones whose channels, on net, support class c.

   ```text
   L^c = ReLU( sum over k of  alpha_k^c * A^k )
   ```
6. **Upsample and overlay.** Resize L^c to the image and draw it with a colormap. The raw coarse map is kept as well. The saved figure and the app both show that coarse map beside the overlay, so the target-layer grid stays visible.

Grad-CAM++ (Chattopadhay et al., 2018) uses the same activations and the same backward pass, and replaces the plain spatial average with weights that depend on higher-order derivatives of the score. The paper's point is coverage when several objects of the chosen class appear in one image. The map is still on the target-layer grid.

Score-CAM (Wang et al., 2020) does not use gradients. Each target-layer activation map is upsampled, normalized, and used as a mask on the input. A forward pass of the masked image produces a class score, and that score becomes the weight of the map. The heatmap is the weighted combination of the maps. The cost is one forward pass per channel, which can be batched. The method avoids saturated gradients by never asking for them, and it pays for that with the extra forwards. The mask itself can create artifacts, so a Score-CAM result is interpreted together with the mask recipe recorded in the config.

### Perturbation methods and gradient methods

Gradient methods differentiate the class score. Grad-CAM, Grad-CAM++, Integrated Gradients, and saliency are in this group. They need a network that can back-propagate, and they typically finish in one backward pass, or in one backward pass per integration step. They can be noisy, and they can be misleading when the gradient is saturated or when the method ignores the parameters that produced it. The sanity check in the evaluation plan exists because a gradient picture can look structured even after the weights have been destroyed (Adebayo et al., 2018).

Perturbation methods change the input and watch the output. Score-CAM masks by activation maps. LIME masks superpixels and fits a local linear model (Ribeiro, Singh, and Guestrin, 2016). KernelSHAP masks features and estimates Shapley values (Lundberg and Lee, 2017). Occlusion, which this project does not schedule for the first version, covers the image with a sliding patch. These methods do not need internal gradients, so they apply to a model that is only available as a forward function. They cost many forward passes. The picture depends on the mask: superpixel size, patch size, and the baseline that replaces a removed region. A fixed seed is part of the run record, because LIME's masks are random.

The app is allowed to show both families on one image. The write-up is not allowed to treat them as interchangeable. A class activation map answers "which spatial features in this layer supported the class." An input attribution answers "which pixels, relative to a baseline, does the gradient credit." A local surrogate answers "which superpixels does a simple model need in order to mimic this one prediction."

## Method comparison

Speed in this table is a qualitative ranking of how the method scales: one backward pass, one backward pass per step, or one forward pass per mask. It is not a measured runtime. Measured seconds are an output of [Runtime](#runtime), and they stay out of the README until a run writes them.

| Method | What it needs | Speed | Strengths | Weaknesses |
| --- | --- | --- | --- | --- |
| Grad-CAM (Selvaraju et al., 2017) | Activations of one convolutional target layer, and gradients of the class logit at that layer. One forward and one backward. | Fast | Class-specific. Works on CNNs that do not use global average pooling. Cheap enough to run on every comparison image. | Only as fine as the target layer. The layer choice changes the map. The ReLU discards evidence against the class. |
| Grad-CAM++ (Chattopadhay et al., 2018) | The same activations and gradients as Grad-CAM, combined with a higher-order weighting of the gradient tensor. | Fast | Aimed at images that contain several instances of the class. Same cost class as Grad-CAM. | Still a coarse target-layer map. The extra weighting is harder to debug when the map looks wrong. |
| Score-CAM (Wang et al., 2020) | Target-layer activations used as input masks. A forward pass for each channel (batchable). No gradient. | Slow | Gradient-free, so saturated gradients cannot dominate the weights. | Cost grows with the channel count. The mask recipe changes the result. Easy to under-batch and then quote a partial timing. |
| Integrated Gradients (Sundararajan et al., 2017) | Gradients of the class logit at the input, along a path from a baseline image to the real image. One forward and one backward per step. | Moderate | Pixel-level. Built to satisfy sensitivity and implementation invariance when the path integral is approximated well. Does not need a convolutional layer. | The baseline is a choice, and a short step count is a coarse integral. The raw map is noisy. This is an input attribution, not a class activation map. |
| Saliency (Simonyan, Vedaldi, and Zisserman, 2014) | The gradient of the class logit with respect to the input pixels. One forward and one backward. | Fast | The simplest pixel-level gradient picture. A useful lower bound on what a gradient method shows. | Noisy. Not a localization map. Adebayo et al. (2018) showed that raw-gradient pictures can stay plausible when the model parameters are randomized. |
| LIME, optional (Ribeiro, Singh, and Guestrin, 2016) | A forward-only model, a superpixel segmentation, and many masked copies of the image. A weighted linear surrogate explains one prediction. | Slow | Model-agnostic. The explanation is a set of superpixels, which is easy to read. | Unstable across seeds and across the superpixel size. Does not use internal features. Too expensive to be the default on every image. |
| SHAP, optional (Lundberg and Lee, 2017) | Depends on the variant. GradientSHAP needs input gradients and a baseline distribution. KernelSHAP needs perturbations and a weighted regression. Captum provides both. | Moderate for GradientSHAP, slow for KernelSHAP | Shapley values give a defined notion of credit across features. | The two variants are different estimators and will be reported separately. KernelSHAP on raw pixels is expensive. A baseline or a feature grouping is still a choice. |

The first implementation wires the five rows above LIME. LIME and SHAP land behind a flag, after the core path has a reportable run. The working SHAP variant, if one is turned on, is named in the config. GradientSHAP and KernelSHAP are never collapsed into a single "SHAP" number.

## Models and datasets

### VisionClassify CIFAR-10 CNN

The first model is the owner's CIFAR-10 CNN from VisionClassify. The owner reports 95.71% test accuracy for that CNN, and [VecScope](https://github.com/cendysaputra/VecScope) cites the same external figure. That number was not measured in this repository. It is recorded here so the loader has a named target. It is not an ExplainVision result, and [README.md](README.md) does not repeat it. It enters the README only if a script in `eval/` scores the checkpoint this repository actually loads and writes the metrics JSON.

The VisionClassify repository has continued since that CNN. The loader will pin the architecture, the weight file, and the commit, in `configs/cifar10_cnn.yaml`. It will not follow a later default checkpoint of a different architecture. Preprocessing is whatever that checkpoint was trained with, stored in the same config. CIFAR-10 images are 32×32 (Krizhevsky, 2009; [dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)). A 32×32 network is not fed a tensor prepared for an ImageNet model.

The CIFAR-10 test set is the split this project will re-score, because it is the model's benchmark and it is small enough to run on a CPU. The test images are also the pool for the failure gallery.

### ImageNet torchvision models

The second family is pretrained CNNs from torchvision, starting with ResNet-50 (He et al., 2016) and one EfficientNet (Tan and Le, 2019). The config records the constructor name and the weights enum, for example a `ResNet50_Weights` value. Preprocessing is that enum's own `transforms()` method, called rather than retyped, so the resize and the normalization cannot drift from the checkpoint. The test that locks this is a comparison against `weights.transforms()` on a fixed image.

ImageNet (Russakovsky et al., 2015; [https://www.image-net.org](https://www.image-net.org)) is the source of those weights. This repository does not download the ImageNet training set in order to explain a photograph. A full ImageNet validation run is optional and is a separate config. Vendor-reported ImageNet accuracy may be discussed in this plan, next to the torchvision version and the model-card URL, and labeled as the vendor's figure. It is not copied into the README.

### Shared rules

- Class names displayed in the app are the names of the loaded model. A CIFAR-10 label is never shown as an ImageNet label, or the reverse.
- The class score that explanations differentiate is the logit.
- Heatmaps are resampled to the image the user sees, and the UI also shows the model's native input so a coarse map is not mistaken for a pixel-level one.
- Datasets and checkpoints are gitignored. A run directory may commit metrics, logs, and small figures. It does not commit weights.

## Evaluation plan

Explanation quality is the thing being measured. Classification accuracy is reported only as context for the loaded checkpoint, and only from a run, as the models section says. The explanation metrics below do not say that a model is accurate. They say whether the heatmap tracks the model's score, whether the heatmap depends on the model's weights, and what the method costs.

### Deletion and insertion AUC

Deletion and insertion follow Petsiuk, Das, and Saenko (2018). An earlier perturbation curve, most-relevant pixels first, is Samek et al. (2017). This project reports the insertion and deletion areas, not a renamed variant.

On one image, for one explained class:

1. Build the heatmap with the method under test.
2. Rank pixels from most important to least important by that heatmap.
3. **Deletion.** Start from the real image. Replace pixels with a constant baseline in that order. After each step, record the softmax probability of the explained class. The area under that curve, with the x-axis scaled to the fraction of pixels removed, is the deletion AUC. A lower area means the probability fell sooner, which is the direction that supports the heatmap having found pixels the model was using.
4. **Insertion.** Start from a blurred copy of the image. Write original pixels back in the same importance order. Record the same probability. A higher insertion AUC is the matching direction.

The constant used for deletion, the blur kernel used for insertion, the fraction grid, and which class was explained are config fields. The working defaults are: deletion fills removed pixels with the dataset mean (zeros in a mean-subtracted tensor), insertion uses a Gaussian blur whose kernel size lives in the config, and the grid is twenty evenly spaced fractions. A change to any of these is a new run, not a silent edit.

The explained class for the headline curve is the predicted class. On labeled images the true class is computed as a second curve and stored separately. The two are not averaged together. The reported figure is the mean AUC over the image list, with the per-image values kept in the JSON. One striking image is not the result.

These scores compare methods on one fixed model and one fixed preprocessing. A higher insertion AUC is not a claim that the method is faithful in every other sense. A method can score well here and still fail the sanity check.

### Sanity check

The required check is the model parameter randomization test from Adebayo, Gilmer, Muelly, Goodfellow, Hardt, and Kim (2018).

- Take the trained model and a fixed list of images. The list is named in the config before the maps are inspected.
- For each method, compute the heatmap with the trained weights.
- Reinitialize parameters in a cascade from the top of the network downward, using the stages the architecture already has. For a ResNet that is the final linear layer, then `layer4`, `layer3`, `layer2`, and `layer1`. The CIFAR-10 CNN lists its own stages in the config once the architecture is known. The random seed is recorded.
- At each cascade step, compute the heatmap again with the same code path.
- Compare each randomized map to the trained map with the Spearman rank correlation of the two flattened heatmaps. Save the two images next to the number. Adebayo et al. show that a similarity score can hide a map that still looks like an edge picture of the input.

A method whose maps stay similar as the weights are destroyed is insensitive to the model. The README says so next to the curve. This plan does not set a numeric pass threshold in advance. A threshold chosen before any map has been seen would be another unmeasured number. The curve and the saved images are the result.

The paper's second check, training on randomly permuted labels, is optional. It requires a retrained model and is a separate experiment. It is not part of the first reportable sanity check.

### Pointing game

The pointing game (Zhang et al., 2016) is optional, and only where a ground-truth region exists. A hit is the single maximum of the heatmap falling inside that region. The score is hits divided by hits plus misses, on images that have a region. The working default adds no pixel slack around the region. If a tolerance is set, it is a config field and the JSON records it.

CIFAR-10 does not ship regions. The ImageNet classification split used to train these classifiers does not either. Until a set with boxes or masks is actually evaluated, the pointing game is recorded as not run. A number from a paper that used a different dataset is not a substitute, and it is not written into the README as an ExplainVision score.

### Runtime

Runtime is wall-clock time per image, batch size one, on one device, after a short warmup that is not included in the measurement. Every method in the comparison sees the same image list. The JSON stores the per-image times, the median, and a simple spread, plus the device string. Score-CAM, LIME, and KernelSHAP are expected to be slower for the reasons in the comparison table. That expectation is not a number, and it is not quoted as one.

A timing run is reportable only when every method in that config finished the full image list. A method that was stopped early stays in the log and stays out of the README.

### What a reportable run contains

Every run that might be quoted lives in `eval/runs/<run-id>/`:

- `config.yaml` — the resolved config, including the model, the weight source, the target layer, the class score (logit), the baseline, the step counts, and the seed.
- `metrics.json` — the metric values, the image ids, the model name, the config path, the git revision, and `"reportable": true` only when the run finished on a clean git tree.
- `train.log` or `eval.log` — the console log of the script that wrote the JSON.
- `environment.txt` — Python, PyTorch, torchvision, pytorch-grad-cam, Captum, OpenCV, matplotlib, and the git revision.
- The figures the metrics refer to, including sanity-check pairs and a sample of deletion curves.

`eval/` at the repository root holds the scripts a person runs. `src/explainvision/metrics/` holds the functions those scripts call. A hand-edited JSON file is not a result. A notebook cell is not a result. Notebooks under `notebooks/` are for exploration.

### Evidence rule

Every number in [README.md](README.md) comes from a reproducible script in `eval/` that was actually run, with the log and the metrics JSON saved, and with the model and config versions recorded in that JSON. A number that was not produced this way is not written into the README. That includes estimates, partial runs, validation scores presented as test scores, timings from a single warm-up iteration, and figures copied from memory or from a model card.

External figures may appear in this plan, labeled as external. The 95.71% CIFAR-10 figure above is one of them. They are not ExplainVision measurements, and they do not move into the README unless this repository reproduces them under the rule.

## Streamlit app

The demo is a single Streamlit app at `app/streamlit_app.py`. It imports preprocessing and explainers from `src/explainvision/`. It does not keep a second copy of those functions.

### Inputs

- Upload a PNG, JPEG, or WEBP image.
- Choose the model: the pinned CIFAR-10 CNN, ResNet-50, or the configured EfficientNet.
- Choose one or more methods. The default selection is Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, and saliency. LIME and SHAP are off unless the installed extras include them, and the sidebar says when they are unavailable.
- Choose the target layer. The default is the layer in the model config. CAM methods that are pointed at a layer with no spatial grid stop with a message that names the layer. Integrated Gradients and saliency ignore the layer and the page says they attribute the input.
- Choose k for the top-k list. The working default is 5, and k cannot exceed the number of classes.
- Choose the class to explain: the top prediction by default, any class in the top-k, or the true label when the image is opened from a labeled manifest.

### What the page shows

- The original image, and beside it the tensor the model scores, rendered back into an image so the preprocessing is visible.
- The top-k classes as bars, with confidences from a softmax over that model's classes. A confidence is a score inside that label set.
- For each selected method, the raw heatmap and the overlay, in a row, so the methods can be compared. The raw heatmap is shown at its native grid and not only after upsampling.
- The overlay colormap and opacity. The working default colormap is `turbo`. Opacity is a slider. Both displayed values are the ones in the config unless the user moves them.
- A footer with the checkpoint id, the config name, the preprocessing name, the target layer, and the git revision.

### Model-failure gallery

A second tab reads a manifest written by `eval/`. The manifest lists misclassified images: file or dataset index, true label, predicted label, and the predicted confidence. The script that writes it walks the CIFAR-10 test set in dataset order, keeps a failure where the top-1 class differs from the label, and stops at a per-class cap. The working default cap is four images per true class. The cap is applied before any heatmap is drawn.

For each manifest row the tab shows the image, the true label, the predicted label, and the heatmaps for both the predicted class and the true class. The page does not add images that the manifest does not list. If the manifest is missing, the tab says that the gallery appears after that script has been run, and it does not invent example failures.

### Before a checkpoint exists

Until the weight file is available, the app states that predictions and heatmaps need a checkpoint. It does not show placeholder class bars. Class names, when a checkpoint is loaded, are the names of that checkpoint.

## Repository structure

```text
ExplainVision/
├── README.md                         # public summary; numbers only from eval runs
├── PLAN.md                           # this plan
├── docs/
│   ├── pipeline.svg                  # static pipeline figure
│   └── pipeline.png
├── configs/                          # cifar10_cnn.yaml, resnet50.yaml, efficientnet.yaml
├── data/                             # CIFAR-10 and local images (gitignored)
├── notebooks/                        # one image, several maps; not a source of README numbers
├── src/explainvision/
│   ├── models/                       # VisionClassify CNN, torchvision wrappers
│   ├── explainers/                   # Grad-CAM family, Captum, optional LIME/SHAP
│   ├── metrics/                      # deletion, insertion, randomization, pointing game, runtime
│   └── utils/                        # preprocessing, overlays, manifests
├── eval/                             # scripts that write logs and metrics JSON
│   └── runs/                         # one directory per reportable run
├── app/
│   └── streamlit_app.py
└── tests/                            # shapes, a known-peak Grad-CAM, preprocessing, reportable flag
```

`src/explainvision/metrics/` holds functions. `eval/` holds the scripts a person runs. The README may quote a number only when one of those scripts wrote it.

## Milestones

Checked tasks are done in this repository. A later phase starts when the previous phase's tasks are done. A phase that produces a measurement is done when the matching directory under `eval/runs/` exists and the evidence rule has been applied.

### Phase 0 — Documents and skeleton

- [x] Write README.md, with the pipeline figure and a Results section that states no scores yet.
- [x] Write this plan.
- [x] Add `docs/pipeline.svg` and `docs/pipeline.png`.
- [ ] Add the package layout under `src/explainvision/` and an install file (`pyproject.toml` or `requirements.txt`) that pins the planned runtime.
- [ ] Add a `.gitignore` for virtual environments, `data/`, checkpoints, and run artifacts that are not metrics.
- [ ] Add the MIT `LICENSE` file.

### Phase 1 — Models and data

- [ ] Pin the VisionClassify CIFAR-10 CNN: architecture, weight file, and commit, in `configs/cifar10_cnn.yaml`.
- [ ] Load that CNN and run one forward pass in a test.
- [ ] Load torchvision ResNet-50 and one EfficientNet from named weights enums.
- [ ] Use each weights enum's own transforms for ImageNet models, with a test against `weights.transforms()` on a fixed image.
- [ ] Script the CIFAR-10 download into `data/`, and leave the files gitignored.
- [ ] Tests for output shape and for finite logits on a dummy batch.

### Phase 2 — Explainers

- [ ] Define one interface: image tensor, model, class index, optional layer, returns an image-sized heatmap and the raw map.
- [ ] Grad-CAM, Grad-CAM++, and Score-CAM through pytorch-grad-cam, reading the configured target layer.
- [ ] Integrated Gradients and saliency through Captum, differentiating the logit.
- [ ] Record the baseline and the integration step count in the config.
- [ ] Optional LIME and one named Captum SHAP variant, off unless enabled.
- [ ] A test on a tiny convolutional network: a bright square on one side of a blank image, Grad-CAM mass on that side.
- [ ] A test that a CAM method refuses a layer with no spatial grid.

### Phase 3 — Evaluation and the failure gallery

- [ ] Deletion and insertion AUC, with the baseline, the grid, and the explained class stored in the JSON.
- [ ] Cascading parameter randomization, Spearman correlation, and the saved map pairs.
- [ ] Runtime per method on a fixed image list, warmup excluded.
- [ ] Pointing game implemented and left unreported until a set with regions is evaluated.
- [ ] An `eval/` entry point that writes the log, `metrics.json`, `environment.txt`, and the resolved config.
- [ ] `"reportable": true` only for a finished run from a clean git revision.
- [ ] The misclassification manifest for the failure gallery, capped per true class, written before heatmaps are drawn.
- [ ] Run the CIFAR-10 CNN evaluation and archive the directory. The README stays free of scores until that archive exists, and then it quotes the JSON.

### Phase 4 — Streamlit

- [ ] Upload, model choice, method choice, layer, class, and top-k.
- [ ] Original image, model input, top-k bars, raw heatmaps, and overlays, side by side.
- [ ] Failure-gallery tab bound to the manifest, including the empty-manifest state.
- [ ] Footer with checkpoint, config, layer, and git revision.
- [ ] A checkpoint-missing state that does not invent class bars.

### Phase 5 — Write-up

- [ ] Fill the README results from `metrics.json` only, each number linked to its run directory.
- [ ] State, in words, any method that failed the randomization check.
- [ ] Leave ImageNet vendor accuracy and the historical 95.71% figure out of the README unless this repository reproduced them.

## Risks and mitigations

| Risk | What we will do |
| --- | --- |
| The class score is the softmax probability, and the gradients shrink on confident predictions. | The config field is `score: logit`. A test checks that the scalar passed to backward is the logit of the chosen class. |
| The target layer is the classifier, or a layer with no spatial grid, and the map is meaningless. | CAM methods require a 4-dimensional activation. The module name is in the config, and a test resolves it. |
| The CIFAR-10 CNN is swapped for a later VisionClassify checkpoint of a different architecture. | The config pins the commit and the weight file. The run JSON repeats both. |
| Preprocessing in the app drifts from preprocessing in `eval/`, so the demo explains a different tensor than the metrics used. | One function per model, called by both. ImageNet models call `weights.transforms()` rather than a retyped copy. |
| The overlay's upsampling hides how coarse Grad-CAM is. | The UI and the saved figures show the raw map beside the overlay. |
| A 32×32 heatmap is read as if it were a high-resolution segmentation. | The UI shows the native input next to the upload and names the grid size. |
| Score-CAM, LIME, or KernelSHAP is stopped early and a partial timing is quoted. | A reportable timing covers the full image list for every method in the config. |
| LIME's random masks move the explanation between runs. | The seed is a config field and is copied into `metrics.json`. |
| The Integrated Gradients baseline or step count changes the map and the change is invisible. | Both are config fields. A different baseline is a different run id. |
| A heatmap looks convincing and does not depend on the weights. | The randomization cascade is required before a method is described as faithful in the README. A method that fails is still shown, and the failure is stated. |
| Deletion AUC is presented as classification accuracy. | The results text names the metric, the direction (lower or higher), the model, and the run id. Accuracy, when it exists, is a separate field. |
| The failure gallery is a few images chosen because the heatmap looks dramatic. | Membership is the manifest, written from dataset order and a per-class cap before heatmaps are computed. |
| The pointing game is quoted on CIFAR-10, which has no regions. | The metric records `not_run` until a set with regions is evaluated. |
| Dataset files or a weight file with its own license are committed under MIT. | `data/` and checkpoints are gitignored. The README says those licenses stay with their owners. |
| A score is typed into the README from memory, from a notebook, or from a model card. | The evidence rule is the gate. Each ExplainVision number in the README names the run id that produced it. |

## References

- Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *IEEE International Conference on Computer Vision*, 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391).
- Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader, and Vineeth N. Balasubramanian. "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks." *IEEE Winter Conference on Applications of Computer Vision*, 2018. [arXiv:1710.11063](https://arxiv.org/abs/1710.11063).
- Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, and Xia Hu. "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks." *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 2020. [arXiv:1910.01279](https://arxiv.org/abs/1910.01279).
- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. "Axiomatic Attribution for Deep Networks." *International Conference on Machine Learning*, 2017. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365).
- Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." *ICLR Workshop*, 2014. [arXiv:1312.6034](https://arxiv.org/abs/1312.6034).
- Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. "Sanity Checks for Saliency Maps." *Advances in Neural Information Processing Systems*, 2018. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292).
- Vitali Petsiuk, Abir Das, and Kate Saenko. "RISE: Randomized Input Sampling for Explanation of Black-box Models." *British Machine Vision Conference*, 2018. [arXiv:1806.07421](https://arxiv.org/abs/1806.07421).
- Wojciech Samek, Alexander Binder, Grégoire Montavon, Sebastian Lapuschkin, and Klaus-Robert Müller. "Evaluating the Visualization of What a Deep Neural Network Has Learned." *IEEE Transactions on Neural Networks and Learning Systems*, 2017.
- Jianming Zhang, Sarah Adel Bargal, Zhe Lin, Jonathan Brandt, Xiaohui Shen, and Stan Sclaroff. "Top-down Neural Attention by Excitation Backprop." *European Conference on Computer Vision*, 2016. [arXiv:1608.00507](https://arxiv.org/abs/1608.00507).
- Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. "Learning Deep Features for Discriminative Localization." *IEEE Conference on Computer Vision and Pattern Recognition*, 2016. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150).
- Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016. [arXiv:1602.04938](https://arxiv.org/abs/1602.04938).
- Scott M. Lundberg and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 2017. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
- Mingxing Tan and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *International Conference on Machine Learning*, 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946).
- Alex Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." Technical report, University of Toronto, 2009. Dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html).
- Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. "ImageNet Large Scale Visual Recognition Challenge." *International Journal of Computer Vision*, 2015. Dataset: [https://www.image-net.org](https://www.image-net.org).
- Jacob Gildenblat and contributors. pytorch-grad-cam. [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
- Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, and Orion Reblitz-Richardson. "Captum: A unified and generic model interpretability library for PyTorch." 2020. [https://captum.ai/](https://captum.ai/). [arXiv:2009.07896](https://arxiv.org/abs/2009.07896).
