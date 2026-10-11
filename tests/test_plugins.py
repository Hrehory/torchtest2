"""The plugin system and the built-in plugins (the framework tests skip if it is not installed)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtest2
from torchtest2 import ModelTester, Plugin, Status, TestSuite, check_model, get_plugin


class TwoHeads(nn.Module):
    """forward(x, y) in training returns a dict of losses; forward(x) returns predictions."""

    def __init__(self, use_aux: bool = True):
        super().__init__()
        self.body = nn.Linear(8, 16)
        self.cls = nn.Linear(16, 3)
        self.aux = nn.Linear(16, 1)
        self.use_aux = use_aux

    def forward(self, x, y=None):
        h = F.relu(self.body(x))
        if y is None:
            return self.cls(h)
        losses = {"cls": F.cross_entropy(self.cls(h), y)}
        if self.use_aux:
            losses["aux"] = self.aux(h).pow(2).mean()
        return losses


class LossDictPlugin(Plugin):
    """A plugin written by a user for their model family."""

    name = "loss_dict"

    def forward(self, model, inputs, targets):
        return model(inputs, targets)

    def loss(self, outputs, targets, loss_fn):
        return sum(outputs.values())

    def configure_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-2)


@pytest.fixture
def data():
    torch.manual_seed(0)
    return torch.randn(16, 8), torch.randint(0, 3, (16,))


def test_custom_plugin(data):
    report = check_model(TwoHeads(), *data, plugin=LossDictPlugin())
    assert report.ok, str(report)
    assert report.info["plugin"] == "loss_dict"
    assert report["loss"].message.endswith("(loss_dict plugin)")
    assert report["optimizer_coverage"].status is Status.PASS  # the plugin's optimizer is used

    report = check_model(TwoHeads(use_aux=False), *data, plugin=LossDictPlugin)  # a class works
    assert report["gradient_flow"].details["no_grad"] == ["aux.weight", "aux.bias"]


def test_plugin_equals_explicit_hooks(data):
    via_plugin = check_model(TwoHeads(use_aux=False), *data, plugin=LossDictPlugin())
    via_hooks = check_model(
        TwoHeads(use_aux=False),
        *data,
        forward_fn=lambda m, x, y: m(x, y),
        loss_fn=lambda losses, y: sum(losses.values()),
    )
    assert [r.status for r in via_plugin] != []
    assert via_plugin["gradient_flow"].details == via_hooks["gradient_flow"].details


def test_explicit_arguments_win_over_the_plugin(data):
    model = TwoHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    report = check_model(model, *data, plugin=LossDictPlugin(), optimizer=optimizer)
    assert report["params_change"].status is Status.FAIL  # lr=0: the explicit optimizer was used
    report = check_model(
        model,
        *data,
        plugin=LossDictPlugin(),
        forward_fn=lambda m, x, y: {"cls": F.cross_entropy(m(x), y)},  # explicit forward
    )
    assert report["gradient_flow"].details["no_grad"] == ["aux.weight", "aux.bias"]


def test_plugin_lookup_by_name_and_registration(data, monkeypatch):
    class EntryPoint:
        name = "loss_dict"

        @staticmethod
        def load():
            return LossDictPlugin

    monkeypatch.setattr(
        "torchtest2.plugins.entry_points",
        lambda group: [EntryPoint()] if group == "torchtest2.plugins" else [],
    )
    assert isinstance(get_plugin("loss_dict"), LossDictPlugin)
    assert check_model(TwoHeads(), *data, plugin="loss_dict").ok
    with pytest.raises(ValueError, match="unknown plugin 'nope'; available: .*lightning"):
        get_plugin("nope")
    with pytest.raises(TypeError, match="not a torchtest2 plugin"):
        get_plugin(object())


def test_plugin_without_batch_dimension(data):
    class NoBatch(Plugin):
        batch_dim = None

    report = check_model(nn.Linear(8, 3), data[0], plugin=NoBatch())
    for name in ("output_varies", "batch_independence"):
        assert report[name].status is Status.SKIP
        assert "no batch dimension" in report[name].message
    assert report.info["batch size"] is None
    assert report["input_dependency"].status is Status.PASS
    assert check_model(nn.Linear(8, 3), data[0], batch_dim=None)["batch_independence"].message == (
        "the inputs have no batch dimension (batch_dim=None)"
    )


def test_plugin_batch_dim_and_explicit_override():
    class SeqFirst(Plugin):
        batch_dim = 1

    x = torch.randn(10, 4, 8)  # (time, batch, features)
    report = check_model(nn.LSTM(8, 16), x, plugin=SeqFirst())
    assert report.info["batch size"] == 4
    assert report["batch_independence"].status is Status.PASS
    report = check_model(nn.LSTM(8, 16), x, plugin=SeqFirst(), batch_dim=0)  # explicit wins
    assert report.info["batch size"] == 10


def test_model_tester_and_test_suite_with_plugin(data):
    class Split(LossDictPlugin):
        def split_batch(self, batch):
            return {"inputs": batch["x"], "targets": batch["y"]}

    batch = {"x": data[0], "y": data[1]}
    tester = ModelTester(TwoHeads(), None, None, batch, plugin=Split())
    assert isinstance(tester.optimizer, torch.optim.Adam)  # from configure_optimizer
    tester.test_parameters_change()
    tester.test_batch_independence()
    assert tester.check().ok
    results = TestSuite(TwoHeads(use_aux=False), None, None, batch, plugin=Split()).test_suite()
    assert results["test_parameters_change"] is False
    assert results["test_model_output_nan"] is True


def test_model_tester_check_uses_its_settings(data):
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    model[0].requires_grad_(False)
    tester = ModelTester(
        model,
        F.cross_entropy,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        list(data),
        nontrain_params=list(model[0].named_parameters(prefix="0")),
        expected_loss=0.5,
        expected_output_shape=(16, 3),
    )
    report = tester.check()
    assert report["frozen_params"].status is Status.PASS
    assert report["output_shape"].status is Status.PASS
    assert report["overfit_batch"].details["losses"][-1] <= 0.5  # expected_loss -> overfit_target


# ------------------------------------------------------------------------------------------------
# built-in plugins, on the real frameworks


def test_builtin_plugins_are_registered():
    assert {"torchvision_detection", "huggingface", "lightning"} <= set(
        torchtest2.plugin_discovery()
    )


def test_torchvision_detection_plugin():
    detection = pytest.importorskip("torchvision.models.detection")
    torch.manual_seed(0)
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=3, min_size=96, max_size=96
    )
    images = [torch.rand(3, 96, 96) for _ in range(2)]
    targets = [
        {"boxes": torch.tensor([[10.0, 20.0, 60.0, 90.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[30.0, 30.0, 90.0, 80.0]]), "labels": torch.tensor([2])},
    ]
    report = check_model(
        model,
        images,
        targets,
        plugin="torchvision_detection",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        overfit_steps=5,
    )
    assert report.ok, str(report)
    assert report["gradient_flow"].status is Status.PASS
    assert report["batch_independence"].status is Status.SKIP


def test_huggingface_plugin():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=3,
    )
    model = transformers.BertForSequenceClassification(config)
    ids = torch.randint(1, 100, (8, 12))
    mask = torch.ones_like(ids)
    mask[::2, 8:] = 0
    inputs = {"input_ids": ids, "attention_mask": mask}
    labels = torch.randint(0, 3, (8,))
    report = check_model(model, inputs, labels, plugin="huggingface", overfit_steps=50)
    assert report.ok, str(report)
    assert report["loss"].message.endswith("(huggingface plugin)")
    assert "labels" not in report["input_dependency"].message
    missing_labels = check_model(model, inputs, plugin="huggingface", overfit_steps=0)
    assert "pass the labels as targets" in missing_labels["loss"].message


def test_lightning_plugin(data):
    lightning = pytest.importorskip("lightning")

    class LitClassifier(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))
            self.unused = nn.Linear(3, 3)

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self(x), y)
            self.log("train_loss", loss)  # works without a Trainer
            return {"loss": loss}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)  # BUG: self.unused missing
            return {"optimizer": optimizer}

    report = check_model(LitClassifier(), *data, plugin="lightning")
    assert report["loss"].status is Status.PASS
    assert report["optimizer_coverage"].status is Status.FAIL  # configure_optimizers was used
    assert set(report["optimizer_coverage"].details["missing"]) == {"unused.weight", "unused.bias"}


def test_huggingface_plugin_with_tuple_outputs():
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=3,
    )
    config.return_dict = False  # the model returns tuples
    model = transformers.BertForSequenceClassification(config)
    ids = torch.randint(1, 100, (4, 8), generator=torch.Generator().manual_seed(0))
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    report = check_model(
        model, inputs, torch.tensor([0, 1, 2, 0]), plugin="huggingface", overfit_steps=5
    )
    assert report["loss"].status is Status.PASS, str(report)
