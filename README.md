<h1 align="center">torchtest2</h1>
<p align="center"><b>Find the silent bugs in your PyTorch model with one function call.</b><br>
Pre-train sanity checks and post-train behavioural tests for PyTorch.</p>

<p align="center">
<a href="https://pypi.org/project/torchtest2/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchtest2?cacheSeconds=3600"></a>
<a href="https://pypi.org/project/torchtest2/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/torchtest2?cacheSeconds=3600"></a>
<img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

Most bugs in deep learning code don't crash. A layer you forgot to call, an optimizer that was
created before you swapped the classification head, an LSTM that treats the batch as time, or an
attention mask that is ignored: the code runs, the loss goes down a bit, and you find out a week
of GPU time later (if ever) that the model is worse than it should be.

`torchtest2` runs a battery of checks on **one batch** and tells you what is wrong:

<!-- readme-test: skip -->
```python
import torchtest2

report = torchtest2.check_model(model, inputs, targets, loss_fn=loss_fn, optimizer=optimizer)
print(report)
```

It works with MLPs, CNNs, RNNs, transformers, and multi-input models taking tensors in
tuples, dicts or dataclasses, including Hugging Face-style models that return their own loss;
plugins adapt families of models whose training call differs (detection models, Lightning). You
don't need to change your model, and the check **restores the PyTorch training state**:
parameters, buffers, optimizer state, gradients, train/eval flags, uninitialised lazy modules,
the PyTorch, Python and NumPy global RNG states and your input tensors are the same afterwards.
Plain Python attributes that your own `forward` changes (e.g. a call counter) are not restored.

---

**Contents**

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [What is checked](#what-is-checked)
4. [Bug gallery: what torchtest2 catches](#bug-gallery-what-torchtest2-catches)
5. [Inputs, outputs and losses](#inputs-outputs-and-losses)
   ([special training forward passes](#models-with-a-special-training-forward-pass))
6. [Plugins: reusable adapters](#plugins-reusable-adapters)
   ([built-in](#built-in-plugins), [writing](#writing-a-plugin), [sharing](#sharing-a-plugin))
7. [Using torchtest2 in pytest](#using-torchtest2-in-pytest)
8. [Configuration reference](#configuration-reference)
9. [Working with the report](#working-with-the-report)
10. [Tensor assertions](#tensor-assertions)
11. [Post-train behavioural tests](#post-train-behavioural-tests)
12. [Class-based API: `ModelTester` and `TestSuite`](#class-based-api-modeltester-and-testsuite)
13. [How it works and limitations](#how-it-works-and-limitations)
14. [Development](#development)
15. [Background and credits](#background-and-credits)

---

## Installation

```bash
pip install torchtest2
```

Requirements: Python ≥ 3.10 and PyTorch ≥ 2.1. There are no other dependencies.

## Quick start

Here is a classifier with a typical bug: a `LayerNorm` that is defined but never used.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtest2


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.norm = nn.LayerNorm(64)  # defined...
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.encoder(x))  # ...but never used


torch.manual_seed(0)
model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x, y = torch.randn(16, 32), torch.randint(0, 10, (16,))

report = torchtest2.check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer)
print(report)
```

```text
torchtest2 report for Classifier  (parameters: 2,890, trainable: 2,890, device: cpu, batch size: 16, objective: loss_fn)
  PASS  forward             output: Tensor[16, 10] float32
  SKIP  output_shape        no output_shape given
  PASS  output_finite       no NaN/inf in the outputs
  SKIP  output_range        no output_range given
  PASS  output_varies       outputs differ between samples
  PASS  eval_determinism    eval-mode forward passes are reproducible
  PASS  loss                loss = 2.323 (loss_fn)
  PASS  initial_loss        initial loss 2.323 is close to ln(10) = 2.303
  PASS  optimizer_coverage  the optimizer covers every trainable parameter
  FAIL  gradient_flow       2/6 trainable parameter(s) get no gradient from the loss_fn (unused in forward, or the graph is cut by .detach()/.item()/no_grad): norm.weight, norm.bias
  PASS  gradient_magnitude  total gradient norm 0.949
  FAIL  params_change       2/6 trainable parameter(s) did not change after an optimizer step: norm.weight, norm.bias
  SKIP  frozen_params       no frozen parameters (requires_grad=False or frozen=[...])
  PASS  input_dependency    all 1 input tensor(s) influence the output
  PASS  batch_independence  outputs of a sample do not depend on other samples in the batch (eval mode)
  PASS  overfit_batch       loss 2.323 -> 0.2244 in 85 step(s)
  SKIP  gpu                 require_gpu=False
  11 passed, 0 warnings, 2 failed, 4 skipped
```

Each check ends in one of four statuses:

| Status | Meaning |
|--------|---------|
| `PASS` | nothing suspicious |
| `WARN` | suspicious, but can be legitimate: read the message |
| `FAIL` | almost certainly a bug |
| `SKIP` | not applicable (e.g. no loss given), or disabled |

The minimum call is `check_model(model, inputs)`. Without a loss function, the loss-based checks
are skipped and the gradient checks use a random projection of the outputs instead (see
[below](#inputs-outputs-and-losses)).

## What is checked

The checks run in this order. If `forward` fails, everything that needs a forward pass is skipped.

| Check | What it does | Typical bug it catches |
|-------|--------------|------------------------|
| `forward` | runs `model(inputs)` in eval mode | shape mismatches, wrong input format |
| `output_shape` | compares the output shape with `output_shape=(None, 10)` | wrong head size, a missing `squeeze`/`flatten` |
| `output_finite` | looks for NaN / inf in all outputs | `log(0)`, division by zero, overflow |
| `output_range` | checks outputs against `output_range=(0, 1)` | a missing `sigmoid`, or applying it twice |
| `output_varies` | *warns* if every sample gets the same output | input wiped out, collapsed activations, broadcasting bugs |
| `eval_determinism` | *warns* if two eval-mode passes differ | `F.dropout(x)` without `training=self.training`, `torch.rand` in `forward` |
| `loss` | the loss is a finite scalar connected to the graph | `reduction="none"`, `.item()`/`.detach()` inside the loss |
| `initial_loss` | cross-entropy at init ≈ `ln(num_classes)` | over-confident initialisation, label leakage |
| `optimizer_coverage` | every trainable parameter is in the optimizer | optimizer created before a layer was added or replaced |
| `gradient_flow` | every trainable parameter gets a non-zero gradient | unused layers, graph cut by `.detach()`/`.data`/`no_grad` |
| `gradient_magnitude` | gradients are finite and not exploding (relative to the parameter's norm) | NaN gradients (e.g. `sqrt` at 0), bad init scale |
| `params_change` | every trainable parameter changes after one optimizer step | any of the above, `lr=0`, zero gradients with Adam |
| `frozen_params` | frozen parameters stay fixed | a "frozen" backbone that is still trained |
| `input_dependency` | every input, and every feature of float inputs, influences the output | wrong slicing, ignored inputs/masks, a variable reused by mistake |
| `batch_independence` | changing one sample doesn't change the outputs of the others | `batch_first` mix-ups, `view` instead of `transpose`, softmax over dim 0 |
| `overfit_batch` | the loss on one batch drops to 10 % of its initial value (or to `overfit_target`), with finite gradients at every step | anything that prevents learning: lr, loss/target mismatch, capacity; NaN gradients that appear only after some training |
| `gradcheck` | (opt-in) `torch.autograd.gradcheck` on a float64 copy of the model | a custom `autograd.Function` with a wrong `backward()` |
| `gpu` | (opt-in) CUDA is available and the model is on it | training silently on the CPU |

## Bug gallery: what torchtest2 catches

Each example below is a real bug pattern. The report lines are real output. The snippets share
this setup:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtest2 import check_model

torch.manual_seed(0)
x, y = torch.randn(16, 32), torch.randint(0, 10, (16,))
```

### 1. The optimizer was created before the head was replaced

A classic fine-tuning bug. The new head is never trained.

```python
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1000))  # "pretrained" model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model[2] = nn.Linear(64, 10)  # new head for 10 classes

report = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer)
```

```text
FAIL  optimizer_coverage  2 trainable parameter(s) are not in the optimizer: 2.weight, 2.bias. The optimizer holds 2 tensor(s) that are not (or no longer) parameters of this model - were layers replaced after creating it, or was it created for another model instance?
FAIL  params_change       2/4 trainable parameter(s) did not change after an optimizer step: 2.weight, 2.bias
```

### 2. The computation graph is cut

`.data`, `.detach()`, `.item()`, a NumPy round-trip or a stray `torch.no_grad()` stop the
gradient. Everything before the cut stays at its random initialisation.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 64)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        features = self.backbone(x).data  # legacy .data silently detaches
        return self.head(F.relu(features))


report = check_model(Net(), x, y, loss_fn=F.cross_entropy)
```

```text
FAIL  gradient_flow       2/4 trainable parameter(s) get no gradient from the loss_fn (unused in forward, or the graph is cut by .detach()/.item()/no_grad): backbone.weight, backbone.bias
```

### 3. `batch_first` mix-ups in RNNs and attention

`nn.LSTM`, `nn.GRU`, `nn.MultiheadAttention` and `nn.Transformer*` default to
`batch_first=False`. If you feed them `(batch, time, features)`, they treat the batch as the
sequence: samples read each other's data, and nothing crashes.

```python
class Encoder(nn.Module):
    def __init__(self, dim=16, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)  # batch_first=False by default!
        self.out = nn.Linear(dim, dim)

    def forward(self, x):  # x: (batch, tokens, dim)
        h, _ = self.attn(x, x, x)
        return self.out(h)


report = check_model(Encoder(), torch.randn(4, 6, 16))
```

```text
FAIL  batch_independence  changing sample 0 changed the outputs of 3/3 other sample(s), e.g. [1, 2, 3]: the model mixes information across the batch (view/reshape instead of transpose/permute, a batch_first mix-up, softmax or normalisation over the batch dimension?)
```

`check_model(nn.LSTM(8, 16), torch.randn(4, 12, 8))` fails the same way. With
`batch_first=True` both pass. The same check catches `x.view(t, b, f)` used where
`x.transpose(0, 1)` was meant, and `softmax`/normalisation over `dim=0`.

### 4. Part of the input is ignored

```python
class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user = nn.Linear(16, 8)
        self.item = nn.Linear(16, 8)

    def forward(self, x):  # x = [user features | item features]
        u, i = x[:, :16], x[:, 16:]
        return (self.user(u) * self.item(u)).sum(-1)  # BUG: item tower gets user features


report = check_model(TwoTower(), x)
```

```text
FAIL  input_dependency    16/32 features of inputs (per-sample shape (32,)) have no influence on the output, e.g. at [(16,), (17,), (18,)] (wrong slice/split, a variable reused by mistake?)
```

### 5. The attention mask is ignored (dict inputs, token ids)

Integer inputs such as token ids and masks are probed by shuffling them across the batch, so the
values stay valid.

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab=1000, dim=32, classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.head = nn.Linear(dim, classes)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids)
        return self.head(h.mean(dim=1))  # BUG: padding is averaged in


ids = torch.randint(1, 1000, (8, 20))
mask = torch.ones_like(ids)
mask[::2, 10:] = 0  # every other sequence is padded
ids[mask == 0] = 0
inputs = {"input_ids": ids, "attention_mask": mask}  # a dict is passed as model(**inputs)
labels = torch.randint(0, 3, (8,))

report = check_model(TextClassifier(), inputs, labels, loss_fn=F.cross_entropy)
```

```text
FAIL  input_dependency    inputs['attention_mask'] has no influence on the output (wrong slice/split, a variable reused by mistake?)
```

### 6. The "frozen" backbone is still being trained

Name the parameters you expect to be frozen with `frozen=` (exact names or prefixes such as
`"backbone"`). Parameters with `requires_grad=False` are always checked.

```python
model = nn.Sequential(nn.Sequential(nn.Linear(32, 64), nn.ReLU()), nn.Linear(64, 10))
for p in model.parameters():  # BUG: meant model[1].parameters()
    p.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

report = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer, frozen=["0"])
```

```text
FAIL  frozen_params       2/2 parameter(s) expected to be frozen changed after an optimizer step: 0.0.weight, 0.0.bias (requires_grad still True? still in the optimizer? modified in-place in forward()?)
```

### 7. NaN gradients behind a clean forward pass

```python
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 8)

    def forward(self, x):
        z = self.fc(x)
        # pairwise euclidean distances; d/dx sqrt(x) is infinite at 0 (the diagonal)
        return (z[:, None] - z[None, :]).pow(2).sum(-1).sqrt()


report = check_model(Embedder(), x)
```

```text
PASS  output_finite       no NaN/inf in the outputs
FAIL  gradient_magnitude  NaN/inf gradients in 2 parameter(s): fc.weight, fc.bias (log/sqrt/division of zero or negative values, overflow in half precision?)
```

### 8. Over-confident initialisation

An untrained classifier should predict roughly uniformly, so its cross-entropy should be about
`ln(num_classes)`. A much higher value means the logits start far too large, which usually wastes
the first part of training.

```python
model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 10))
nn.init.normal_(model[2].weight, std=1.0)  # instead of the default ~U(-1/16, 1/16)

report = check_model(model, x, y, loss_fn=F.cross_entropy)
```

```text
WARN  initial_loss        initial loss 10.510 is much larger than ln(10) = 2.303 expected from an untrained classifier; the network is over-confident at init (scale down the last layer's init, check input normalisation)
```

### 9. Dropout that stays on at inference

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(32, 64), nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(F.dropout(F.relu(self.fc1(x)), p=0.2))  # training=True by default!


report = check_model(Net(), x, y, loss_fn=F.cross_entropy)
```

```text
WARN  eval_determinism    two forward passes in eval mode differ (output); randomness in eval mode - e.g. F.dropout(x) without training=self.training, or torch.rand in forward()
```

### 10. Loss problems

```python
report = check_model(Net(), x, y, loss_fn=nn.CrossEntropyLoss(reduction="none"))
```

```text
FAIL  loss                the loss must be a scalar, got shape (16,) (reduction='none' without .mean()?)
```

A loss that goes through `.item()` or NumPy fails with *"the loss (...) is not connected to the
autograd graph"*. A learning rate that is far too small fails `overfit_batch`:

```python
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
report = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer)
```

```text
FAIL  overfit_batch       the loss did not decrease on a single batch in 200 steps (2.271 -> best 2.269); check the learning rate, the optimizer and that targets match outputs
```

### 11. NaN gradients that appear only after some training

`torch.where` does not protect the backward pass. The gradient of the branch that was *not*
selected is still computed, and if it is NaN, `0 * NaN = NaN` poisons the parameter. Below, a
noise-injection layer learns its variance. The optimizer drives the variance towards zero and,
a few steps later, past it: the forward pass stays finite, the gradients do not. The first steps
look fine, so `gradient_magnitude` passes; `overfit_batch` checks the gradients at every step.

```python
class NoisyNet(nn.Module):
    """Gaussian noise injection with a learnable variance."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)
        self.var = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        logits = self.fc(x)
        if self.training:
            std = torch.where(self.var > 0, torch.sqrt(self.var), torch.zeros_like(self.var))
            logits = logits + std * torch.randn_like(logits)
        return logits


model = NoisyNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
report = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer)
```

```text
FAIL  overfit_batch       gradients became NaN/inf at step 31 (loss 0.3996) in var; rerun with detect_anomaly=True to find the operation
```

With `detect_anomaly=True` the check runs under `torch.autograd.detect_anomaly`. It then names
the operation, and PyTorch prints the forward-pass traceback of the line that created it:

```python
report = check_model(model, x, y, loss_fn=F.cross_entropy, optimizer=optimizer, detect_anomaly=True)
```

```text
FAIL  overfit_batch       torch.autograd.detect_anomaly: Function 'SqrtBackward0' returned nan values in its 0th output. (the warning printed above shows the forward-pass traceback of the offending operation)
```

### 12. A custom `autograd.Function` with a wrong `backward()`

Hand-written backward passes are easy to get subtly wrong: the model still trains, just badly.
`gradcheck=True` compares them with finite differences, using a float64 copy of the model and
the first two samples.

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return x * grad  # BUG: the derivative of x**2 is 2 * x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc(Square.apply(x))


report = check_model(Net(), x, y, loss_fn=F.cross_entropy, gradcheck=True)
```

```text
FAIL  gradcheck           analytical and numerical gradients w.r.t. the inputs disagree (float64); a custom autograd.Function has a wrong backward(), or the model is not differentiable here
```

## Inputs, outputs and losses

**Inputs** are one batch, with the batch dimension first:

| You pass | The model is called as |
|----------|------------------------|
| a tensor `x` | `model(x)` |
| a tuple / list `(a, b)` | `model(a, b)` |
| a dict `{"input_ids": ..., "attention_mask": ...}` | `model(input_ids=..., attention_mask=...)` |

If your model takes a single list argument, wrap it: `inputs=(my_list,)`. A dataclass is passed
as a single argument too. Inputs and targets are moved to the device of the model's parameters
automatically, and your tensors are never modified: every tensor in tuples, lists, dicts,
namedtuples and dataclasses is copied (tensors stored in other objects are not).

**Outputs** can be a tensor, a tuple, a dict, a namedtuple, a dataclass or a Hugging Face
`ModelOutput`. The checks inspect every tensor in them.

**The loss** is picked in this order:

1. `loss_fn(outputs, targets)`, or `loss_fn(outputs)` when `targets` is `None`. Any callable
   works, so a multi-output model can use its own function:

   ```python
   def loss_fn(outputs, targets):
       logits, boxes = outputs
       return F.cross_entropy(logits, targets["labels"]) + F.l1_loss(boxes, targets["boxes"])
   ```

2. A loss computed by the model itself (`outputs.loss` or `outputs["loss"]`), which is the
   Hugging Face convention. Put the labels into the inputs:

   ```python
   class HFStyle(nn.Module):
       def __init__(self):
           super().__init__()
           self.emb = nn.Embedding(1000, 32)
           self.head = nn.Linear(32, 3)

       def forward(self, input_ids, attention_mask, labels=None):
           m = attention_mask.unsqueeze(-1).float()
           logits = self.head((self.emb(input_ids) * m).sum(1) / m.sum(1))
           loss = F.cross_entropy(logits, labels) if labels is not None else None
           return {"loss": loss, "logits": logits}


   report = check_model(HFStyle(), {**inputs, "labels": labels})
   assert report.info["objective"] == "output.loss"
   ```

   The same works for a real `transformers` model:
   `check_model(model, {**tokenizer(texts, return_tensors="pt", padding=True), "labels": labels})`.

3. No loss at all. The loss-dependent checks (`loss`, `initial_loss`, `overfit_batch`) are
   skipped, and the gradient checks use a fixed random projection of all outputs,
   `sum(output * noise)`. Unlike `output.sum()`, this cannot be constant: `softmax(x).sum()` is
   always 1 and would report "no gradient" for every layer.

**The optimizer** is the one you train with, so `optimizer_coverage` and `overfit_batch` test
your real setup, including the learning rate. If you pass none, `Adam(lr=1e-3)` over the
trainable parameters is used and `optimizer_coverage` is skipped.

### Models with a special training forward pass

Some models are not trained through `model(inputs)`. Two hooks cover them:

- `forward_fn(model, inputs, targets)` is used for the training-mode passes (loss,
  gradients, optimizer steps, overfitting); whatever it returns is what `loss_fn` receives as
  its first argument - logits, or, for a model that computes its own losses, those losses;
- `eval_forward_fn(model, inputs) -> outputs` is used for the eval-mode passes (outputs, input
  dependency, batch independence).

For example, a model that takes the targets in training mode and returns a dict of losses:

```python
class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(32, 64)
        self.cls = nn.Linear(64, 10)
        self.aux = nn.Linear(64, 1)

    def forward(self, x, y=None):
        h = F.relu(self.body(x))
        if y is None:  # inference
            return self.cls(h)
        return {"cls": F.cross_entropy(self.cls(h), y), "aux": self.aux(h).pow(2).mean()}


report = check_model(
    MultiTask(),
    x,
    y,
    forward_fn=lambda model, x, y: model(x, y),  # training-mode call: returns the dict of losses
    loss_fn=lambda losses, y: sum(losses.values()),  # receives that dict -> one scalar
)
assert report.ok
```

**Sequence-first tensors.** If the batch is not the first dimension, e.g. `(time, batch,
features)` for an `nn.LSTM` without `batch_first=True`, pass `batch_dim=1`. If the outputs have
their batch in another dimension than the inputs, e.g. a sequence classifier with inputs
`(time, batch, features)` and outputs `(batch, classes)`, pass `output_batch_dim` too
(`batch_dim=1, output_batch_dim=0`). If the inputs have no batch dimension at all (e.g. a list
of images of different sizes), pass `batch_dim=None`: the batch checks are then skipped.

If you need the same hooks in many tests, or for a whole family of models, write a plugin.

## Plugins: reusable adapters

A plugin bundles everything torchtest2 needs to know about a family of models: how to run it in
training and in evaluation mode, where the loss comes from, the batch dimension and, optionally,
the optimizer. Pass it as `plugin=` to `check_model`, `ModelTester` or `TestSuite`, as an
instance, a class or a name. Explicit arguments always take precedence over the plugin.

### Built-in plugins

| Name | For | What it does |
|------|-----|--------------|
| `"torchvision_detection"` | torchvision detection models (Faster/Mask R-CNN, RetinaNet, FCOS, SSD, ...) | train: `model(images, targets)` and the sum of the loss dict; eval: `model(images)`; the images are a list, so there is no batch dimension |
| `"huggingface"` | Hugging Face `transformers` models | train: `model(**inputs, labels=targets)` and `outputs.loss`; eval: `model(**inputs)` |
| `"lightning"` | PyTorch Lightning `LightningModule`s | train: `module.training_step((inputs, targets), 0)` and the loss it returns; the optimizer comes from `configure_optimizers()` |

None of them imports its framework, so torchtest2 still depends on nothing but PyTorch.

**torchvision detection:**

```python
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

detector = fasterrcnn_mobilenet_v3_large_fpn(
    weights=None, weights_backbone=None, num_classes=3, min_size=128, max_size=128
)
images = [torch.rand(3, 128, 128) for _ in range(2)]
targets = [
    {"boxes": torch.tensor([[10.0, 20.0, 60.0, 90.0]]), "labels": torch.tensor([1])},
    {"boxes": torch.tensor([[30.0, 30.0, 100.0, 120.0]]), "labels": torch.tensor([2])},
]
report = check_model(
    detector,
    images,
    targets,
    plugin="torchvision_detection",
    optimizer=torch.optim.SGD(detector.parameters(), lr=0.01),
    overfit_steps=30,
)
```

All checks that apply run, including gradient flow through all 192 parameter tensors;
`output_varies` and `batch_independence` are skipped because a list of images has no batch
dimension.

**Hugging Face.** The labels are passed as targets, so the input checks probe only the real
inputs (token ids, attention mask):

```python
from transformers import BertConfig, BertForSequenceClassification

config = BertConfig(
    vocab_size=1000,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    num_labels=3,
)
bert = BertForSequenceClassification(config)  # or .from_pretrained(...) with a tokenizer
report = check_model(bert, inputs, labels, plugin="huggingface", overfit_steps=50)
assert report.ok
```

**PyTorch Lightning.** The plugin supports common `LightningModule` patterns: training passes
call `training_step(batch, 0)`, the optimizer comes from `configure_optimizers()` (the first one),
and `self.log()` calls outside a trainer are harmless. It does not reproduce the Trainer loop
(strategies, precision plugins, manual optimization, multiple optimizers, schedulers, gradient
accumulation, callbacks):

```python
import lightning as L


class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        self.aux_head = nn.Linear(64, 10)  # added later...

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)  # ...but not optimised


report = check_model(LitClassifier(), x, y, plugin="lightning")
```

```text
FAIL  optimizer_coverage  2 trainable parameter(s) are not in the optimizer: aux_head.weight, aux_head.bias. Was the optimizer created before these layers were added/replaced?
```

Pass `LightningPlugin(make_batch=...)` if your `training_step` expects a different batch
structure than `(inputs, targets)`.

### Writing a plugin

Subclass `torchtest2.Plugin` and override what your models need. Every method has a default that
matches plain `check_model` behaviour. This is how torchtest2 uses a plugin - the value returned
by `forward` is handed, unchanged, to `loss`:

<!-- readme-test: skip -->
```python
# every training-mode pass (loss, gradients, optimizer steps, overfitting):
train_out = plugin.forward(model, inputs, targets)  # whatever your model returns in training
loss = plugin.loss(train_out, targets, loss_fn)  # turn it into one scalar -> loss.backward()
# every eval-mode pass (outputs, input dependency, batch independence):
outputs = plugin.eval_forward(model, inputs)
```

[docs/architecture.md](https://github.com/Hrehory/torchtest2/blob/master/docs/architecture.md) shows this as a diagram, next to the life cycle of a
`check_model` call and which checks run in which mode.

For `MultiTask` (above), the training-mode call `model(x, y)` returns a **dict of losses**, not
logits, so the plugin's `loss` receives that dict and sums it. The parameter can be named after
what it holds:

```python
from torchtest2 import Plugin


class MultiTaskPlugin(Plugin):
    """For our multi-task models: forward(x, y) returns a dict of losses in training."""

    name = "multitask"
    batch_dim = 0  # None if the inputs have no batch dimension
    output_batch_dim = None  # set if the outputs have the batch in another dimension

    def forward(self, model, inputs, targets):  # training-mode passes
        return model(inputs, targets)  # MultiTask in training: {"cls": ..., "aux": ...}

    def eval_forward(self, model, inputs):  # eval-mode passes
        return model(inputs)  # MultiTask at inference: logits

    def loss(self, losses, targets, loss_fn):  # losses: what forward() returned
        return losses["cls"] + losses["aux"]  # loss_fn (the user's, if any) is not needed here

    def configure_optimizer(self, model):  # used when no optimizer is passed
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def split_batch(self, batch):  # only for ModelTester/TestSuite(batch=...)
        return {"inputs": batch[0], "targets": batch[1]}


report = check_model(MultiTask(), x, y, plugin=MultiTaskPlugin())
assert report.info["plugin"] == "multitask"
```

### Sharing a plugin

To make a plugin available by name, for your team or on PyPI, register it under the
`torchtest2.plugins` entry-point group in the `pyproject.toml` of the package that contains it:

```toml
[project.entry-points."torchtest2.plugins"]
multitask = "my_company_ml.testing:MultiTaskPlugin"
```

After `pip install`, `check_model(model, x, y, plugin="multitask")` works everywhere, and
`torchtest2.plugin_discovery()` lists the built-in and installed plugins. A name is resolved
among the built-in plugins first, then only the entry point of that name is imported, so a
broken unrelated plugin cannot break your tests (`plugin_discovery()` skips it with a
`RuntimeWarning`). Built-in names cannot be overridden.

**Plugins written for the 2021 development version** keep working: `torchtest2.PluginBase`
maps `train_inference` -> `forward`, `test_inference` -> `eval_forward` and
`compute_loss(loss_criterion, outputs, targets)` -> `loss`. Objects that only implement those
methods without subclassing it are accepted too, and so is the 2021 `plugins` entry-point group.

## Using torchtest2 in pytest

`assert_model_ok` runs all checks and raises `ModelCheckError` (an `AssertionError`) listing the
failed checks and the whole report:

<!-- readme-test: skip -->
```python
import pytest
import torch
import torch.nn.functional as F
import torchtest2

from my_project.models import build_model  # your code


@pytest.mark.parametrize("arch", ["resnet18", "vit_tiny", "my_new_idea"])
def test_architecture_is_trainable(arch):
    torch.manual_seed(0)
    model = build_model(arch, num_classes=10)
    x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 10, (8,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    torchtest2.assert_model_ok(
        model,
        x,
        y,
        loss_fn=F.cross_entropy,
        optimizer=optimizer,
        output_shape=(None, 10),
        overfit_steps=100,
    )
```

A failing test prints something like:

```text
E   torchtest2.exceptions.ModelCheckError: 2 check(s) did not pass: gradient_flow, params_change
E
E   torchtest2 report for MyNewIdea  (...)
E     PASS  forward             output: Tensor[8, 10] float32
E     ...
```

Some useful variations:

<!-- readme-test: skip -->
```python
# treat warnings as failures
torchtest2.assert_model_ok(model, x, y, loss_fn=F.cross_entropy, strict=True)

# skip checks that don't apply, e.g. a contrastive model that deliberately compares samples
torchtest2.assert_model_ok(model, x, skip=["batch_independence"])

# keep the report and assert on single checks
report = torchtest2.check_model(model, x, y, loss_fn=F.cross_entropy)
assert report["gradient_flow"].status is torchtest2.Status.PASS
```

All exceptions derive from `torchtest2.TorchTestError`, which is an `AssertionError`, so pytest
reports them as test failures rather than errors.

## Configuration reference

<!-- readme-test: skip -->
```python
torchtest2.check_model(
    model, inputs, targets=None, *,
    loss_fn=None,               # loss_fn(outputs, targets) -> scalar tensor
    optimizer=None,             # your optimizer; default Adam(lr=1e-3)
    output_shape=None,          # e.g. (None, 10); None/-1 = any size
    output_range=None,          # e.g. (0, 1) or (None, 0); inclusive
    frozen=(),                  # names/prefixes of parameters that must not change
    overfit_steps=200,          # 0 disables overfit_batch
    overfit_ratio=0.1,          # pass when loss <= ratio * initial loss
    overfit_target=None,        # ...or when loss <= this absolute value
    detect_anomaly=False,       # run overfit_batch under torch.autograd.detect_anomaly
    grad_norm_threshold=1e3,    # gradient norm / parameter norm that triggers a warning
    gradcheck=False,            # enable the gradcheck check (custom autograd.Functions)
    require_gpu=False,          # enable the gpu check
    plugin=None,                # a Plugin (instance, class or name) for a family of models
    forward_fn=None,            # forward_fn(model, inputs, targets) for train-mode passes
    eval_forward_fn=None,       # eval_forward_fn(model, inputs) for eval-mode passes
    batch_dim=0,                # batch dimension (the plugin's by default); None = no batch
    output_batch_dim=None,      # batch dimension of the outputs, if not batch_dim
    skip=(),                    # names of checks to skip, see torchtest2.CHECKS
    seed=0,                     # seed of the private RNG the checks use
    verbose=False,              # print the report
) -> torchtest2.Report
```

Tips:

- **Large models.** Use a small configuration of the same architecture (fewer layers, smaller
  width) and a small batch (4–16 samples). Most bugs don't depend on size, and every check needs
  only a few forward/backward passes, plus up to `overfit_steps` for `overfit_batch`.
- **`overfit_batch` too slow or too strict?** Lower `overfit_steps`, raise `overfit_ratio`, or
  `skip=["overfit_batch"]`. A `WARN` there means the loss went down, just not far enough.
- **Batch of at least 2 distinct samples** is needed for `output_varies` and `batch_independence`.

## Working with the report

```python
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
report = torchtest2.check_model(model, x, y, loss_fn=F.cross_entropy)

report.ok  # True if nothing FAILed (warnings allowed)
report.failures, report.warnings  # lists of CheckResult
report["gradient_flow"].status  # Status.PASS / WARN / FAIL / SKIP
report["gradient_flow"].details  # e.g. {"no_grad": ["norm.weight", "norm.bias"], ...} on failure
report["overfit_batch"].details["losses"]  # the loss curve of the overfitting run
report.info  # parameter counts, device, batch size, objective (and the plugin, if any)
report.to_dict()  # JSON-friendly, e.g. for wandb.log / mlflow.log_dict
report.raise_on_failure()  # raise ModelCheckError if anything failed
```

## Tensor assertions

Small building blocks for your own tests. Each raises a `TorchTestError` subclass.

```python
from torchtest2 import (
    assert_all_finite,
    assert_never_nan,
    assert_never_inf,
    assert_in_range,
    assert_shape,
    assert_gpu_available,
)

probs = torch.sigmoid(torch.randn(4, 3))
assert_in_range(probs, 0, 1)  # RangeException otherwise
assert_in_range(probs, 0, 1, inclusive=False)
assert_shape(probs, (None, 3))  # ConsistencyException otherwise
assert_all_finite(probs)  # NaNTensorException / InfTensorException
```

```text
>>> assert_never_nan(torch.tensor([1.0, float("nan")]))
torchtest2.exceptions.NaNTensorException: Tensor contains 1 NaN value(s)!
>>> assert_in_range(torch.tensor([0.5, 1.5]), 0, 1)
torchtest2.exceptions.RangeException: Tensor values are outside of the expected range [0, 1]: observed min = 0.5, max = 1.5.
```

## Post-train behavioural tests

Accuracy on a held-out set doesn't tell you *how* a model fails. Behavioural testing, popularised
for NLP by [CheckList](https://arxiv.org/abs/2005.04118), writes explicit expectations about a
trained model. torchtest2 has the three CheckList test types. Each one runs the model in eval
mode without gradients and raises an `AssertionError` subclass that lists the failing samples in
`.failing_indices`.

```python
from torchtest2 import assert_minimum_functionality, assert_invariance, assert_directional


class ToySentiment(nn.Module):
    """Stands in for a trained model: positive iff the features sum to more than 0."""

    def forward(self, x):
        score = x.sum(-1)
        return torch.stack([-score, score], dim=-1)  # logits for (negative, positive)


model = ToySentiment()
easy_x = torch.tensor([[2.0, 1.0, 0.5], [-3.0, -1.0, 0.2]])
easy_y = torch.tensor([1, 0])  # hand-labelled, unambiguous cases
positive = lambda out: out.softmax(-1)[:, 1]

# MFT - minimum functionality: simple cases the model must get right
assert_minimum_functionality(model, easy_x, easy_y, min_accuracy=1.0)

# INV - invariance: a label-preserving change must not change the prediction
#   (images: small shifts or flips; text: swapping names or locations)
assert_invariance(model, easy_x, lambda t: t.flip(-1), output_fn=lambda out: out.argmax(-1))

# DIR - directional expectation: a change with a known effect must move the score the right way
#   (sentiment: appending "I loved it" must not decrease P(positive))
delta = assert_directional(model, easy_x, lambda t: t + 1, output_fn=positive, expect="increase")

try:
    assert_directional(model, easy_x, lambda t: t - 1, output_fn=positive, expect="increase")
except torchtest2.exceptions.DirectionalExpectationException as e:
    print(e)  # the score was expected to increase ... for 2/2 sample(s), e.g. [0, 1]
    print(e.failing_indices)  # [0, 1]
```

With a real text model, wrap the tokenizer and the model so that it takes raw strings (a sketch):

<!-- readme-test: skip -->
```python
def add_positive_phrase(batch_texts):
    return [t + " I loved it." for t in batch_texts]


class SentimentWrapper(nn.Module):  # makes a text pipeline look like model(texts)
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer

    def forward(self, texts):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        return self.model(**enc).logits


assert_directional(
    SentimentWrapper(model, tok),
    (reviews,),
    lambda b: (add_positive_phrase(b[0]),),
    output_fn=lambda logits: logits.softmax(-1)[:, 1],
    expect="increase",
)
```

## Class-based API: `ModelTester` and `TestSuite`

The original class-based API is still available, rebuilt on the same engine. There are four
ways to run the checks:

| | Runs | On failure |
|-|------|------------|
| `check_model(...)` | all 18 checks | returns a `Report` (`.raise_on_failure()` raises) |
| `ModelTester(...).test_*()` | one test | raises a specific exception |
| `ModelTester(...).test_suite()` | the tests enabled in `test_suite_config` | raises the first failure |
| `TestSuite(...).test_suite()` | the same tests | runs them all, prints a ✓/× summary and returns `{test: passed}` |

Every method runs on a snapshot, so the model is restored even when a test fails.

```python
from torchtest2 import ModelTester

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
tester = ModelTester(
    model,
    loss_criterion=F.cross_entropy,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    batch=[torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))],
)

tester.test_parameters_change()  # ParametersChangeException
tester.test_input_dependency()  # InputDependencyException
tester.test_batch_independence()  # BatchIndependencyException
tester.test_output_consistency((None, 10))  # ConsistencyException
tester.test_model_output_nan()  # NaNTensorException
tester.test_model_output_inf()  # InfTensorException
tester.test_anomaly(iter_number=10)  # AnomalyDetectionError (detect_anomaly while training)
tester.test_gradcheck()  # GradCheckException (float64 copy of the model)
best_loss = tester.test_overfit(expected_loss=0.05)  # OverfitException / OverfitNaNException
best_loss = tester.overfit_single_batch()  # early stopping via min/max_epochs, n_epochs_stop
tester.test_suite()  # the tests enabled in test_suite_config
tester.get_number_of_parameters()  # 7850
report = tester.check()  # the full check_model report, with the tester's settings
```

`test_suite_config` maps test names to flags. The keys are the names of the `ModelTester` test
methods, and `torchtest2.model_tester.DEFAULT_SUITE_CONFIG` holds the defaults, used when the
config is `None`:

| Key | Default | Key | Default |
|-----|---------|-----|---------|
| `test_gpu_available` | off | `test_model_output_inf` | on |
| `test_parameters_change` | on | `test_input_dependency` | on |
| `test_train_params_change` | off | `test_batch_independence` | on |
| `test_nontrain_params_change` | off | `test_overfit` | off |
| `test_output_consistency` | off | `test_gradcheck` | off |
| `test_model_output_range` | off | `test_anomaly` | on |
| `test_model_output_nan` | on | | |

Keys missing from a config you pass are disabled. The 2021 `master` names (`test_params_change`,
`test_output_range`, `test_nan_values`, `test_inf_values`) are accepted as aliases.
`test_model_output_range` checks *strict* bounds (`low < x < high`), as in 2021, whereas
`check_model(output_range=...)` and `assert_in_range` include the bounds.

`TestSuite` collects all results instead of stopping at the first failure:

```python
from torchtest2 import TestSuite

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
model[1].requires_grad_(False)  # a frozen layer
suite = TestSuite(
    model,
    F.cross_entropy,
    torch.optim.Adam(model.parameters(), lr=1e-2),
    [torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))],
    test_suite_config={"test_parameters_change": True, "test_nontrain_params_change": True},
)
results = suite.test_suite()
```

```text
SUMMARY
  test_parameters_change       ✓
  test_nontrain_params_change  ✓
not performed tests: test_gpu_available, test_train_params_change, test_output_consistency, test_model_output_range, test_model_output_nan, test_model_output_inf, test_input_dependency, test_batch_independence, test_overfit, test_gradcheck, test_anomaly
```

Both classes accept `forward_fn`, `eval_forward_fn`, `batch_dim`, `output_batch_dim` and
`plugin=` (the plugin's
`split_batch` turns `batch` into inputs and targets). `torchtest2.assert_tensor_range(tensor,
value, relation, comparison_condition)` from 2021 is also still available.

## How it works and limitations

Diagrams of what happens inside `check_model`, of the checks by mode and of the plugin calls are
in [docs/architecture.md](https://github.com/Hrehory/torchtest2/blob/master/docs/architecture.md).

- **What is restored.** Before the checks run, torchtest2 takes an in-memory snapshot of the
  model's `state_dict` (kept on the CPU), the buffers it leaves out (registered with
  `persistent=False`), the optimizer state, the existing `.grad` tensors and
  every module's train/eval flag. It restores them before each check and at the end. The checks
  use a forked PyTorch CPU/CUDA RNG (`torch.random.fork_rng`), and the Python `random` and
  NumPy global RNG states are saved and restored too (each check seeds all of them, so compared
  forward passes draw the same random numbers). Inputs are copied. A model with uninitialised
  lazy modules (`nn.LazyLinear`, ...) is checked as an initialised copy and stays uninitialised;
  if you pass an optimizer for it, the model and the optimizer are copied together (an optimizer
  that cannot be copied is rejected with a `TypeError`; create it after one forward pass instead).
  **Not restored:** plain Python attributes or other objects that the model's
  own code changes (counters, caches, lists), MPS/XPU random state, and tensors stored in
  objects other than tuples, lists, dicts, namedtuples and dataclasses. Memory cost: about one
  extra CPU copy of the parameters and buffers, plus one copy of the optimizer state.
- **Deterministic.** The same model, data and `seed` give the same report.
- **Training-mode checks** (`loss`, `gradient_*`, `params_change`, `frozen_params`,
  `overfit_batch`) use `model.train()`. **Inference checks** (`output_*`, `input_dependency`,
  `batch_independence`) use `model.eval()`, so BatchNorm's legitimate batch mixing during
  training is not reported.
- **`batch_independence`** replaces sample 0 with sample 1 and checks that samples 1…N-1 keep
  their outputs. It needs a batch dimension in both inputs and outputs (`batch_dim`, first by
  default, and `output_batch_dim` if the outputs differ). Outputs without it are ignored, and it
  skips when no such output exists or when `batch_dim=None`. If the outputs change structure
  (an output appears or disappears, a dict key or a batched shape changes), it fails. It
  detects *leaks between samples*, not bugs that scramble data *within* a sample, such as a
  wrong head split in attention. On CUDA with TF32 enabled (PyTorch's default for
  convolutions), a coupling is re-checked with TF32 off: TF32 kernels may round a sample
  differently depending on its neighbours, which is a property of the kernel, not of the model.
- **`input_dependency`** uses gradients for float inputs (feature by feature, summed over the
  batch). It rolls integer and bool inputs along the batch dimension, which keeps token ids and
  masks valid, and checks the whole tensor. Features that are zero in every sample count as
  padding. Inputs that do not reach the eval outputs are re-checked through the training
  objective (e.g. detections after non-maximum suppression) and after one optimizer step
  (zero-initialised heads and residual branches, as in torchvision's ViT and many timm models).
  Pixels at the trailing border that a strided convolution or a patch embedding never covers
  (up to a quarter of the image) give a `WARN`.
- **Zero initialisation is common and legitimate**: at initialisation, layers behind a
  zero-initialised layer get exactly zero gradients. If the gradient becomes non-zero after one
  optimizer step, `gradient_flow` and `params_change` pass with a note; a parameter whose
  gradient stays zero gives a `WARN`. A `FAIL` is reserved for parameters that get *no* gradient
  at all, or that do not change although their gradient is non-zero.
- **Stochastic depth and LayerDrop** skip whole layers in some training passes. A parameter
  without a gradient is re-checked with other random seeds; if it gets a gradient in one of
  them, it is reported as a note, not as unused.
- **Heuristics warn, they don't fail.** `initial_loss`, `output_varies`, `eval_determinism`,
  `gradient_magnitude` (exploding) and an incomplete `overfit_batch` produce `WARN`. Use
  `strict=True` if you want them to fail your tests.
- Models whose parameters are spread over several devices work, as long as `model(inputs)` works
  with inputs on the device of the first parameter.

## Development

In a clone of the repository:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # ruff lint + format on every commit
pytest                      # ~10 s on a laptop CPU; includes running every example in this README
mypy                        # type check
```

`pip install -e ".[dev,integrations]"` also installs torchvision, transformers and Lightning, so
that the tests of the built-in plugins and their README examples run instead of being skipped.
See [CONTRIBUTING.md](https://github.com/Hrehory/torchtest2/blob/master/CONTRIBUTING.md).

## Background and credits

torchtest2 started as a re-implementation of
[torchtest](https://github.com/suriyadeepan/torchtest/) (no longer maintained and
GPL-licensed), itself inspired by [mltest](https://github.com/Thenerdstation/mltest) and Chase
Roberts' post
[*MLTest: automatically test neural network models in one function call*](https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d).
Further reading behind the checks:

- Jeremy Jordan, [*Effective testing for machine learning systems*](https://www.jeremyjordan.me/testing-ml/)
- Ribeiro et al., [*Beyond Accuracy: Behavioral Testing of NLP models with CheckList*](https://arxiv.org/abs/2005.04118) (ACL 2020)
- Tilman Krokotsch, [*How to trust your deep learning code*](https://krokotsch.eu/cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html) (the batch-independence test)
- Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/) (verify the loss at init, overfit one batch)

Licensed under the [Apache License 2.0](https://github.com/Hrehory/torchtest2/blob/master/LICENSE.txt).
