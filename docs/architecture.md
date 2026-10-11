# How torchtest2 works

Three diagrams of the library: what `check_model` does around the checks, which checks run the
model in which mode, and how a plugin plugs into that. Each diagram has one source in
[`diagrams/`](diagrams/) (`*.mmd`); `diagrams/build.sh` renders them to PDF for the tutorial
slides and checks that this page embeds the same sources. GitHub and GitLab render
the diagrams below; elsewhere, see the PDFs next to the sources.

## What `check_model` does around the checks

Before the first check, torchtest2 decides how to run the model, copies the inputs and takes a
snapshot of the PyTorch state; every check starts from that snapshot with freshly seeded random
generators, and at the end everything is restored. This is why a call has no effect on the
model, the optimizer, the random generators or your tensors (see "How it works and limitations"
in the README for exactly what is restored).

```mermaid
flowchart TB
    subgraph BEFORE["before the checks"]
        direction LR
        A["check_model(model,<br/>inputs, targets, ...)"] --> B["resolve how to run<br/>the model: arguments,<br/>then plugin, then defaults"]
        B --> C["copy the inputs<br/>(lazy modules:<br/>an initialised copy)"]
        C --> D["snapshot: state_dict,<br/>non-persistent buffers,<br/>optimizer state, .grad,<br/>train/eval flags"]
        D --> E["fork the PyTorch,<br/>Python and NumPy<br/>RNGs"]
    end
    subgraph EACH["for each of the 18 checks"]
        direction LR
        G["restore the snapshot, reseed"] --> H["run the check: PASS / WARN / FAIL / SKIP"]
    end
    subgraph AFTER["after the checks"]
        direction LR
        I["restore the snapshot and the RNGs"] --> J["Report"]
    end
    BEFORE --> EACH --> AFTER
```

## Which checks run the model, and how

Inference checks run the model in eval mode, training checks in train mode; two checks do not run
the model at all. The mode decides which plugin method is used (next diagram).

```mermaid
flowchart LR
    E["<b>eval mode</b><br/><i>plugin.eval_forward</i><br/><br/>forward<br/>output_shape<br/>output_finite<br/>output_range<br/>output_varies<br/>eval_determinism<br/>input_dependency<br/>batch_independence<br/>gradcheck (opt-in)"]
    T["<b>train mode</b><br/><i>plugin.forward + plugin.loss</i><br/><br/>loss<br/>initial_loss<br/>gradient_flow<br/>gradient_magnitude<br/>params_change<br/>frozen_params<br/>overfit_batch"]
    N["<b>no forward pass</b><br/><br/>optimizer_coverage<br/>gpu (opt-in)"]
    E ~~~ T ~~~ N
```

## How a plugin is used

A plugin does not change what is checked, only how the model is run. For every training-mode
check, torchtest2 calls `plugin.forward(model, inputs, targets)` and hands its return value,
unchanged, to `plugin.loss(train_out, targets, loss_fn)`, which turns it into one scalar. That
value can be logits, but also a dict of losses (detection models, multi-task models) or a model
output object (Hugging Face) - whatever your model returns in training. Eval-mode checks call
`plugin.eval_forward(model, inputs)`. Without a plugin, all three default to calling
`model(inputs)` and using your `loss_fn`.

```mermaid
sequenceDiagram
    participant C as check
    participant P as plugin
    participant M as model
    Note over C,M: training-mode checks (loss, gradients, optimizer steps, overfitting)
    C->>P: forward(model, inputs, targets)
    P->>M: model(inputs, targets)
    M-->>P: train_out: logits, a dict of losses, ...
    P-->>C: train_out, unchanged
    C->>P: loss(train_out, targets, loss_fn)
    P-->>C: one scalar
    Note over C: backward(), inspect gradients,<br/>optimizer step
    Note over C,M: eval-mode checks (outputs, input dependency, batch independence)
    C->>P: eval_forward(model, inputs)
    P->>M: model(inputs)
    M-->>C: outputs
```
