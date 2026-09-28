from pathlib import Path

import torch

from recskill import build_model_from_genome, validate_model_forward
from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models.matching.comirec import ComirecDR, ComirecSA
from torch_rechub.models.matching.dssm import DSSM
from torch_rechub.models.matching.dssm_facebook import FaceBookDSSM
from torch_rechub.models.matching.dssm_senet import DSSM as DSSMSENET
from torch_rechub.models.matching.gru4rec import GRU4Rec
from torch_rechub.models.matching.mind import MIND
from torch_rechub.models.matching.narm import NARM
from torch_rechub.models.matching.sasrec import SASRec
from torch_rechub.models.matching.sine import SINE
from torch_rechub.models.matching.stamp import STAMP
from torch_rechub.models.matching.youtube_dnn import YoutubeDNN
from torch_rechub.models.matching.youtube_sbc import YoutubeSBC
from torch_rechub.models.multi_task.aitm import AITM
from torch_rechub.models.multi_task.esmm import ESMM
from torch_rechub.models.multi_task.mmoe import MMOE
from torch_rechub.models.multi_task.ple import PLE
from torch_rechub.models.multi_task.shared_bottom import SharedBottom
from torch_rechub.models.multi_domain.mmoe import MMOE as MultiDomainMMOE
from torch_rechub.models.multi_domain.ple import PLE as MultiDomainPLE
from torch_rechub.models.multi_domain.sharebottom import SharedBottom as MultiDomainSharedBottom
from torch_rechub.models.multi_domain.adasparse import AdaSparse as MultiDomainAdaSparse
from torch_rechub.models.multi_domain.adaptdhm import AdaptDHM as MultiDomainAdaptDHM
from torch_rechub.models.multi_domain.epnet import EPNet as MultiDomainEPNet
from torch_rechub.models.multi_domain.hamur import HamurLarge as MultiDomainHamurLarge
from torch_rechub.models.multi_domain.hamur import HamurSmall as MultiDomainHamurSmall
from torch_rechub.models.multi_domain.m2m import M2M as MultiDomainM2M
from torch_rechub.models.multi_domain.m3oe import M3oE as MultiDomainM3oE
from torch_rechub.models.multi_domain.ppnet import PPNet as MultiDomainPPNet
from torch_rechub.models.multi_domain.sarnet import Sarnet as MultiDomainSarnet
from torch_rechub.models.multi_domain.star import Star as MultiDomainStar
from torch_rechub.models.ranking.afm import AFM
from torch_rechub.models.ranking.autoint import AutoInt
from torch_rechub.models.ranking.bst import BST
from torch_rechub.models.ranking.dcn import DCN
from torch_rechub.models.ranking.dcn_v2 import DCNv2
from torch_rechub.models.ranking.deepffm import DeepFFM, FatDeepFFM
from torch_rechub.models.ranking.deepfm import DeepFM
from torch_rechub.models.ranking.dien import DIEN
from torch_rechub.models.ranking.din import DIN
from torch_rechub.models.ranking.edcn import EDCN
from torch_rechub.models.ranking.fibinet import FiBiNet
from torch_rechub.models.ranking.widedeep import WideDeep


ROOT = Path(__file__).resolve().parents[2]
GENOMES = ROOT / "recskill" / "genomes"


def _sparse_features(vocab_sizes, embedding_dim: int, prefix: str = "f"):
    return [SparseFeature(f"{prefix}_{idx}", vocab_size=vocab_size, embed_dim=embedding_dim) for idx, vocab_size in enumerate(vocab_sizes)]


def _stack_sparse_batch(features, batch_size: int, high: int | None = None):
    tensors = []
    for feature in features:
        max_id = min(feature.vocab_size, high) if high is not None else feature.vocab_size
        tensors.append(torch.randint(0, max_id, (batch_size,)))
    return torch.stack(tensors, dim=1)


def test_dcn_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(1)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 5
    hidden_dims = [7]
    features = [SparseFeature(f"f_{idx}", vocab_size=vocab_size, embed_dim=embedding_dim) for idx, vocab_size in enumerate(vocab_sizes)]
    original = DCN(features=features, n_cross_layers=2, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"})
    original.eval()
    original_batch = {
        feature.name: torch.randint(0, feature.vocab_size, (batch_size,))
        for feature in features
    }
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "dcn_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "flat_input_dim": len(vocab_sizes) * embedding_dim,
            "n_cross_layers": 2,
            "hidden_dims": hidden_dims,
            "fusion_dim": len(vocab_sizes) * embedding_dim + hidden_dims[-1],
        },
    )
    sparse_features = torch.stack([original_batch[feature.name] for feature in features], dim=1)
    ctx = model({"sparse_features": sparse_features})

    assert ctx["logits"].shape == (batch_size, 1)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, {"sparse_features": sparse_features}, required_outputs=["prediction"])
    assert ok, message


def test_din_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(2)
    batch_size = 4
    seq_len = 5
    embedding_dim = 4
    context_feature = SparseFeature("context", vocab_size=19, embed_dim=embedding_dim)
    history_feature = SequenceFeature("hist_item", vocab_size=23, embed_dim=embedding_dim, pooling="concat", padding_idx=0)
    target_feature = SparseFeature("target_item", vocab_size=29, embed_dim=embedding_dim)
    original = DIN(
        features=[context_feature],
        history_features=[history_feature],
        target_features=[target_feature],
        mlp_params={"dims": [8], "dropout": 0.0},
        attention_mlp_params={"dims": [8], "activation": "dice", "use_softmax": False},
    )
    original.eval()
    original_batch = {
        "context": torch.randint(0, context_feature.vocab_size, (batch_size,)),
        "hist_item": torch.randint(1, history_feature.vocab_size, (batch_size, seq_len)),
        "target_item": torch.randint(0, target_feature.vocab_size, (batch_size,)),
    }
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "din_skillified.yaml",
        runtime_params={
            "context_vocab_sizes": [context_feature.vocab_size],
            "history_vocab_sizes": [history_feature.vocab_size],
            "target_vocab_sizes": [target_feature.vocab_size],
            "embedding_dim": embedding_dim,
            "num_history_fields": 1,
            "attention_hidden_dims": [8],
            "din_input_dim": 3 * embedding_dim,
            "hidden_dims": [8],
        },
    )
    batch = {
        "context_features": original_batch["context"].unsqueeze(1),
        "history_features": original_batch["hist_item"].unsqueeze(1),
        "target_features": original_batch["target_item"].unsqueeze(1),
    }
    ctx = model(batch)

    assert ctx["history_embeddings"].shape == (batch_size, 1, seq_len, embedding_dim)
    assert ctx["attention_pooling"].shape == (batch_size, 1, embedding_dim)
    assert ctx["logits"].shape == (batch_size, 1)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_sasrec_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(3)
    batch_size = 4
    seq_len = 6
    vocab_size = 31
    embedding_dim = 5
    seq = SequenceFeature("seq", vocab_size=vocab_size, embed_dim=embedding_dim, pooling="concat", padding_idx=0)
    pos = SequenceFeature("pos", vocab_size=vocab_size, embed_dim=embedding_dim, pooling="concat", shared_with="seq", padding_idx=0)
    neg = SequenceFeature("neg", vocab_size=vocab_size, embed_dim=embedding_dim, pooling="concat", shared_with="seq", padding_idx=0)
    original = SASRec(features=[seq, pos, neg], max_len=seq_len, dropout_rate=0.0, num_blocks=1, num_heads=1)
    original.eval()
    original_batch = {
        "seq": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "pos": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "neg": torch.randint(1, vocab_size, (batch_size, seq_len)),
    }
    original_pos_logits, original_neg_logits = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "sasrec_skillified.yaml",
        runtime_params={
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "max_len": seq_len,
            "num_blocks": 1,
            "num_heads": 1,
        },
    )
    batch = {
        "seq_features": original_batch["seq"],
        "pos_features": original_batch["pos"],
        "neg_features": original_batch["neg"],
    }
    ctx = model(batch)

    assert ctx["seq_embeddings"].shape == (batch_size, seq_len, embedding_dim)
    assert ctx["pos_embeddings"].shape == (batch_size, seq_len, embedding_dim)
    assert ctx["neg_embeddings"].shape == (batch_size, seq_len, embedding_dim)
    assert ctx["sequence_output"].shape == (batch_size, seq_len, embedding_dim)
    assert ctx["pos_logits"].shape == original_pos_logits.shape
    assert ctx["neg_logits"].shape == original_neg_logits.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["pos_logits", "neg_logits"])
    assert ok, message


def test_mmoe_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(4)
    batch_size = 4
    vocab_sizes = [17, 19, 23]
    embedding_dim = 4
    expert_hidden_dims = [8]
    task_types = ["classification", "regression"]
    tower_hidden_dims = [[6], [5]]
    features = [SparseFeature(f"f_{idx}", vocab_size=vocab_size, embed_dim=embedding_dim) for idx, vocab_size in enumerate(vocab_sizes)]
    original = MMOE(
        features=features,
        task_types=task_types,
        n_expert=3,
        expert_params={"dims": expert_hidden_dims, "dropout": 0.0, "activation": "relu"},
        tower_params_list=[
            {"dims": tower_hidden_dims[0], "dropout": 0.0, "activation": "relu"},
            {"dims": tower_hidden_dims[1], "dropout": 0.0, "activation": "relu"},
        ],
    )
    original.eval()
    original_batch = {
        feature.name: torch.randint(0, feature.vocab_size, (batch_size,))
        for feature in features
    }
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "mmoe_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": len(vocab_sizes) * embedding_dim,
            "n_expert": 3,
            "n_task": len(task_types),
            "expert_hidden_dims": expert_hidden_dims,
            "expert_output_dim": expert_hidden_dims[-1],
            "task_types": task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    sparse_features = torch.stack([original_batch[feature.name] for feature in features], dim=1)
    ctx = model({"sparse_features": sparse_features})

    assert ctx["task_representations"].shape == (batch_size, len(task_types), expert_hidden_dims[-1])
    assert ctx["task_outputs"].shape == original_output.shape
    ok, message = validate_model_forward(model, {"sparse_features": sparse_features}, required_outputs=["task_outputs"])
    assert ok, message


def test_deepfm_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(5)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    hidden_dims = [8]
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = DeepFM(deep_features=features, fm_features=features, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"})
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "deepfm_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "flat_input_dim": len(vocab_sizes) * embedding_dim,
            "hidden_dims": hidden_dims,
        },
    )
    sparse_features = torch.stack([original_batch[feature.name] for feature in features], dim=1)
    labels = torch.randint(0, 2, (batch_size,)).float()
    ctx = model({"sparse_features": sparse_features, "labels": labels})

    assert ctx["prediction"].shape == original_output.shape
    assert ctx["logits"].shape == (batch_size, 1)
    assert ctx["loss"].shape == torch.Size([])
    ok, message = validate_model_forward(model, {"sparse_features": sparse_features, "labels": labels}, required_outputs=["prediction", "loss"])
    assert ok, message


def test_widedeep_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(6)
    batch_size = 4
    embedding_dim = 4
    wide_features = _sparse_features([11, 13], embedding_dim, prefix="wide")
    deep_features = _sparse_features([17, 19, 23], embedding_dim, prefix="deep")
    hidden_dims = [8]
    original = WideDeep(wide_features, deep_features, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"})
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in wide_features + deep_features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "widedeep_skillified.yaml",
        runtime_params={
            "wide_vocab_sizes": [feature.vocab_size for feature in wide_features],
            "deep_vocab_sizes": [feature.vocab_size for feature in deep_features],
            "embedding_dim": embedding_dim,
            "wide_input_dim": len(wide_features) * embedding_dim,
            "deep_input_dim": len(deep_features) * embedding_dim,
            "hidden_dims": hidden_dims,
        },
    )
    batch = {
        "wide_features": torch.stack([original_batch[feature.name] for feature in wide_features], dim=1),
        "deep_features": torch.stack([original_batch[feature.name] for feature in deep_features], dim=1),
    }
    ctx = model(batch)

    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_dcnv2_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(7)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    hidden_dims = [7]
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = DCNv2(
        features=features,
        n_cross_layers=2,
        mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
        model_structure="parallel",
        use_low_rank_mixture=True,
        low_rank=3,
        num_experts=2,
    )
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "dcnv2_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "flat_input_dim": len(vocab_sizes) * embedding_dim,
            "n_cross_layers": 2,
            "low_rank": 3,
            "num_experts": 2,
            "hidden_dims": hidden_dims,
            "fusion_dim": len(vocab_sizes) * embedding_dim + hidden_dims[-1],
        },
    )
    batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}
    ctx = model(batch)

    assert ctx["cross_output"].shape == (batch_size, len(vocab_sizes) * embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_edcn_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(8)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    input_dim = len(vocab_sizes) * embedding_dim
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = EDCN(features=features, n_cross_layers=2, mlp_params={"dims": [input_dim, input_dim], "dropout": 0.0, "activation": "relu"})
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "edcn_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "flat_input_dim": input_dim,
            "num_fields": len(vocab_sizes),
            "field_dims": [embedding_dim] * len(vocab_sizes),
            "n_cross_layers": 2,
            "hidden_dims": [input_dim, input_dim],
            "bridge_type": "hadamard_product",
            "use_regulation_module": True,
            "temperature": 1.0,
            "edcn_fusion_dim": input_dim * 3,
        },
    )
    batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}
    ctx = model(batch)

    assert ctx["edcn_stack_output"].shape == (batch_size, input_dim * 3)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_afm_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(9)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = AFM(features, embed_dim=embedding_dim, t=5)
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "afm_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "num_fields": len(vocab_sizes),
            "flat_input_dim": len(vocab_sizes) * embedding_dim,
            "attention_dim": 5,
        },
    )
    batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}
    ctx = model(batch)

    assert ctx["afm_output"].shape == (batch_size, embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_autoint_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(10)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    hidden_dims = [8]
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = AutoInt(
        sparse_features=features,
        dense_features=[],
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "autoint_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "flat_input_dim": len(vocab_sizes) * embedding_dim,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dims": hidden_dims,
        },
    )
    batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}
    ctx = model(batch)

    assert ctx["attention_embeddings"].shape == (batch_size, len(vocab_sizes), embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_fibinet_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(11)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    hidden_dims = [8]
    features = _sparse_features(vocab_sizes, embedding_dim)
    original = FiBiNet(features=features, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"}, reduction_ratio=2)
    original.eval()
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_output = original(original_batch)
    num_pairs = len(vocab_sizes) * (len(vocab_sizes) - 1) // 2

    model = build_model_from_genome(
        GENOMES / "fibinet_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "num_fields": len(vocab_sizes),
            "reduction_ratio": 2,
            "bilinear_type": "field_interaction",
            "fibinet_input_dim": 2 * num_pairs * embedding_dim,
            "hidden_dims": hidden_dims,
        },
    )
    batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}
    ctx = model(batch)

    assert ctx["fibinet_flat"].shape == (batch_size, 2 * num_pairs * embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_deepffm_genomes_dummy_forward_shapes_match_originals():
    torch.manual_seed(12)
    batch_size = 4
    embedding_dim = 4
    hidden_dims = [8]
    num_fields = 3
    linear_features = _sparse_features([20, 22], 1, prefix="linear")
    cross_features = _sparse_features([30, 32, 34], embedding_dim, prefix="cross")
    original_deepffm = DeepFFM(linear_features, cross_features, embed_dim=embedding_dim, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"})
    original_fat = FatDeepFFM(linear_features, cross_features, embed_dim=embedding_dim, reduction_ratio=1, mlp_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"})
    original_deepffm.eval()
    original_fat.eval()
    original_batch = {feature.name: torch.randint(0, min(feature.vocab_size, 5), (batch_size,)) for feature in linear_features + cross_features}
    deepffm_output = original_deepffm(original_batch)
    fat_output = original_fat(original_batch)
    num_pairs = num_fields * (num_fields - 1) // 2
    common_params = {
        "linear_vocab_sizes": [feature.vocab_size for feature in linear_features],
        "cross_vocab_sizes": [feature.vocab_size for feature in cross_features],
        "linear_embedding_dim": 1,
        "embedding_dim": embedding_dim,
        "num_fields": num_fields,
        "ffm_input_dim": num_pairs * embedding_dim,
        "hidden_dims": hidden_dims,
    }
    batch = {
        "linear_features": torch.stack([original_batch[feature.name] for feature in linear_features], dim=1),
        "cross_features": torch.stack([original_batch[feature.name] for feature in cross_features], dim=1),
    }

    deepffm_model = build_model_from_genome(GENOMES / "deepffm_skillified.yaml", runtime_params=common_params)
    deepffm_ctx = deepffm_model(batch)
    assert deepffm_ctx["ffm_interactions"].shape == (batch_size, num_pairs, embedding_dim)
    assert deepffm_ctx["prediction"].shape == deepffm_output.shape

    fat_model = build_model_from_genome(
        GENOMES / "fatdeepffm_skillified.yaml",
        runtime_params={**common_params, "num_field_crosses": num_pairs, "reduction_ratio": 1},
    )
    fat_ctx = fat_model(batch)
    assert fat_ctx["gated_ffm_flat"].shape == (batch_size, num_pairs * embedding_dim)
    assert fat_ctx["prediction"].shape == fat_output.shape


def test_bst_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(13)
    batch_size = 4
    seq_len = 5
    embedding_dim = 4
    context = SparseFeature("context", vocab_size=11, embed_dim=embedding_dim)
    history = SequenceFeature("history", vocab_size=13, embed_dim=embedding_dim, pooling="concat", padding_idx=0)
    target = SparseFeature("target", vocab_size=13, embed_dim=embedding_dim, padding_idx=0)
    original = BST(
        features=[context],
        history_features=[history],
        target_features=[target],
        mlp_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        nhead=1,
        dropout=0.0,
        num_layers=1,
        max_seq_len=seq_len + 1,
    )
    original.eval()
    original_batch = {
        "context": torch.randint(0, context.vocab_size, (batch_size,)),
        "history": torch.randint(1, history.vocab_size, (batch_size, seq_len)),
        "target": torch.randint(1, target.vocab_size, (batch_size,)),
    }
    original_output = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "bst_skillified.yaml",
        runtime_params={
            "context_vocab_sizes": [context.vocab_size],
            "history_vocab_sizes": [history.vocab_size],
            "target_vocab_sizes": [target.vocab_size],
            "embedding_dim": embedding_dim,
            "num_history_fields": 1,
            "sequence_dim": embedding_dim,
            "n_heads": 1,
            "n_layers": 1,
            "max_seq_len": seq_len + 1,
            "bst_input_dim": 3 * embedding_dim,
            "hidden_dims": [8],
        },
    )
    batch = {
        "context_features": original_batch["context"].unsqueeze(1),
        "history_features": original_batch["history"].unsqueeze(1),
        "target_features": original_batch["target"].unsqueeze(1),
    }
    ctx = model(batch)

    assert ctx["bst_sequence"].shape == (batch_size, seq_len + 1, embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction"])
    assert ok, message


def test_dien_genome_dummy_forward_shape_matches_original():
    torch.manual_seed(14)
    batch_size = 4
    seq_len = 5
    embedding_dim = 4
    context = SparseFeature("context", vocab_size=11, embed_dim=embedding_dim)
    target = SparseFeature("target", vocab_size=13, embed_dim=embedding_dim, padding_idx=0)
    history = SequenceFeature("history", vocab_size=13, embed_dim=embedding_dim, pooling="concat", shared_with="target", padding_idx=0)
    neg_history = SequenceFeature("neg_history", vocab_size=13, embed_dim=embedding_dim, pooling="concat", shared_with="target", padding_idx=0)
    original = DIEN(
        features=[context],
        history_features=[history],
        neg_history_features=[neg_history],
        target_features=[target],
        mlp_params={"dims": [8], "dropout": 0.0},
    )
    original.eval()
    original_batch = {
        "context": torch.randint(0, context.vocab_size, (batch_size,)),
        "history": torch.randint(1, history.vocab_size, (batch_size, seq_len)),
        "neg_history": torch.randint(1, neg_history.vocab_size, (batch_size, seq_len)),
        "target": torch.randint(1, target.vocab_size, (batch_size,)),
    }
    original_output, original_aux = original(original_batch)

    model = build_model_from_genome(
        GENOMES / "dien_skillified.yaml",
        runtime_params={
            "context_vocab_sizes": [context.vocab_size],
            "history_vocab_sizes": [history.vocab_size],
            "target_vocab_sizes": [target.vocab_size],
            "embedding_dim": embedding_dim,
            "num_history_fields": 1,
            "dien_input_dim": 3 * embedding_dim,
            "hidden_dims": [8],
        },
    )
    batch = {
        "context_features": original_batch["context"].unsqueeze(1),
        "history_features": original_batch["history"].unsqueeze(1),
        "target_features": original_batch["target"].unsqueeze(1),
    }
    ctx = model(batch)

    assert ctx["interest_evolving"].shape == (batch_size, 1, embedding_dim)
    assert ctx["prediction"].shape == original_output.shape
    assert ctx["aux_loss"].shape == original_aux.shape
    ok, message = validate_model_forward(model, batch, required_outputs=["prediction", "aux_loss"])
    assert ok, message


def test_shared_bottom_ple_esmm_aitm_genome_shapes_match_originals():
    torch.manual_seed(15)
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 4
    input_dim = len(vocab_sizes) * embedding_dim
    features = _sparse_features(vocab_sizes, embedding_dim)
    task_types = ["classification", "regression"]
    tower_hidden_dims = [[6], [5]]
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    sparse_batch = {"sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1)}

    shared_bottom = SharedBottom(
        features=features,
        task_types=task_types,
        bottom_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params_list=[
            {"dims": tower_hidden_dims[0], "dropout": 0.0, "activation": "relu"},
            {"dims": tower_hidden_dims[1], "dropout": 0.0, "activation": "relu"},
        ],
    )
    shared_bottom.eval()
    shared_model = build_model_from_genome(
        GENOMES / "shared_bottom_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "bottom_hidden_dims": [8],
            "bottom_output_dim": 8,
            "n_task": len(task_types),
            "task_types": task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    shared_ctx = shared_model(sparse_batch)
    assert shared_ctx["task_outputs"].shape == shared_bottom(original_batch).shape

    ple = PLE(
        features=features,
        task_types=task_types,
        n_level=1,
        n_expert_specific=1,
        n_expert_shared=1,
        expert_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params_list=[
            {"dims": tower_hidden_dims[0], "dropout": 0.0, "activation": "relu"},
            {"dims": tower_hidden_dims[1], "dropout": 0.0, "activation": "relu"},
        ],
    )
    ple.eval()
    ple_model = build_model_from_genome(
        GENOMES / "ple_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "n_task": len(task_types),
            "n_expert_shared": 1,
            "n_expert_specific": 1,
            "expert_hidden_dims": [8],
            "expert_output_dim": 8,
            "task_types": task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    ple_ctx = ple_model(sparse_batch)
    assert ple_ctx["task_outputs"].shape == ple(original_batch).shape

    user_features = _sparse_features([11, 13], embedding_dim, prefix="user")
    item_features = _sparse_features([17, 19], embedding_dim, prefix="item")
    esmm = ESMM(
        user_features=user_features,
        item_features=item_features,
        cvr_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        ctr_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
    )
    esmm.eval()
    esmm_batch_original = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in user_features + item_features}
    esmm_model = build_model_from_genome(
        GENOMES / "esmm_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "item_vocab_sizes": [feature.vocab_size for feature in item_features],
            "embedding_dim": embedding_dim,
            "tower_input_dim": (len(user_features) + len(item_features)) * embedding_dim,
            "cvr_hidden_dims": [8],
            "ctr_hidden_dims": [8],
        },
    )
    esmm_batch = {
        "user_features": torch.stack([esmm_batch_original[feature.name] for feature in user_features], dim=1),
        "item_features": torch.stack([esmm_batch_original[feature.name] for feature in item_features], dim=1),
    }
    esmm_ctx = esmm_model(esmm_batch)
    assert esmm_ctx["task_outputs"].shape == esmm(esmm_batch_original).shape

    aitm_task_types = ["classification", "classification", "classification"]
    aitm = AITM(
        features=features,
        n_task=len(aitm_task_types),
        bottom_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params_list=[{"dims": [6], "dropout": 0.0, "activation": "relu"} for _ in aitm_task_types],
    )
    aitm.eval()
    aitm_model = build_model_from_genome(
        GENOMES / "aitm_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "n_task": len(aitm_task_types),
            "bottom_hidden_dims": [8],
            "bottom_output_dim": 8,
            "task_types": aitm_task_types,
            "tower_hidden_dims": [[6], [6], [6]],
        },
    )
    aitm_ctx = aitm_model(sparse_batch)
    assert aitm_ctx["task_outputs"].shape == aitm(original_batch).shape


def test_multi_domain_shared_bottom_mmoe_ple_genome_shapes_match_originals():
    torch.manual_seed(26)
    batch_size = 4
    domain_num = 3
    embedding_dim = 4
    base_features = _sparse_features([11, 13, 17], embedding_dim)
    domain_feature = SparseFeature("domain_indicator", vocab_size=domain_num, embed_dim=embedding_dim)
    features = base_features + [domain_feature]
    vocab_sizes = [feature.vocab_size for feature in features]
    input_dim = len(features) * embedding_dim
    domain_task_types = ["classification"] * domain_num
    tower_hidden_dims = [6]
    domain_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    original_batch = {
        feature.name: torch.randint(0, feature.vocab_size, (batch_size,))
        for feature in base_features
    }
    original_batch["domain_indicator"] = domain_ids
    sparse_batch = {
        "sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1),
        "domain_indicator": domain_ids,
    }

    shared_bottom = MultiDomainSharedBottom(
        features=features,
        domain_num=domain_num,
        bottom_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params={"dims": tower_hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    shared_bottom.eval()
    shared_model = build_model_from_genome(
        GENOMES / "multi_domain_shared_bottom_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "bottom_hidden_dims": [8],
            "bottom_output_dim": 8,
            "domain_num": domain_num,
            "domain_task_types": domain_task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    shared_ctx = shared_model(sparse_batch)
    assert shared_ctx["domain_outputs"].shape == (batch_size, domain_num)
    assert shared_ctx["prediction"].shape == shared_bottom(original_batch).shape

    mmoe = MultiDomainMMOE(
        features=features,
        domain_num=domain_num,
        n_expert=3,
        expert_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params={"dims": tower_hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    mmoe.eval()
    mmoe_model = build_model_from_genome(
        GENOMES / "multi_domain_mmoe_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "n_expert": 3,
            "expert_hidden_dims": [8],
            "expert_output_dim": 8,
            "domain_task_types": domain_task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    mmoe_ctx = mmoe_model(sparse_batch)
    assert mmoe_ctx["domain_representations"].shape == (batch_size, domain_num, 8)
    assert mmoe_ctx["prediction"].shape == mmoe(original_batch).shape

    ple = MultiDomainPLE(
        features=features,
        domain_num=domain_num,
        n_level=1,
        n_expert_specific=1,
        n_expert_shared=1,
        expert_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        tower_params={"dims": tower_hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    ple.eval()
    ple_model = build_model_from_genome(
        GENOMES / "multi_domain_ple_skillified.yaml",
        runtime_params={
            "vocab_sizes": vocab_sizes,
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "n_level": 1,
            "n_expert_specific": 1,
            "n_expert_shared": 1,
            "expert_params": {"dims": [8], "dropout": 0.0, "activation": "relu"},
            "expert_output_dim": 8,
            "domain_task_types": domain_task_types,
            "tower_hidden_dims": tower_hidden_dims,
        },
    )
    ple_ctx = ple_model(sparse_batch)
    assert ple_ctx["domain_representations"].shape == (batch_size, domain_num, 8)
    assert ple_ctx["prediction"].shape == ple(original_batch).shape


def test_multi_domain_adasparse_epnet_ppnet_genome_shapes_match_originals():
    torch.manual_seed(27)
    batch_size = 4
    domain_num = 3
    embedding_dim = 4
    domain_ids = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    scenario_features = [SparseFeature("domain_indicator", vocab_size=domain_num, embed_dim=embedding_dim)]
    agnostic_features = _sparse_features([11, 13, 17], embedding_dim, prefix="agn")
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in agnostic_features}
    original_batch["domain_indicator"] = domain_ids
    scenario_tensor = domain_ids.unsqueeze(1)
    agnostic_tensor = torch.stack([original_batch[feature.name] for feature in agnostic_features], dim=1)

    adasparse = MultiDomainAdaSparse(
        sce_features=scenario_features,
        agn_features=agnostic_features,
        form="Fusion",
        epsilon=1e-2,
        alpha=1.0,
        delta_alpha=1e-4,
        mlp_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
    )
    adasparse.eval()
    adasparse_model = build_model_from_genome(
        GENOMES / "multi_domain_adasparse_skillified.yaml",
        runtime_params={
            "scenario_vocab_sizes": [domain_num],
            "agnostic_vocab_sizes": [feature.vocab_size for feature in agnostic_features],
            "embedding_dim": embedding_dim,
            "scenario_dim": embedding_dim,
            "agnostic_dim": len(agnostic_features) * embedding_dim,
            "hidden_dims": [8],
            "form": "Fusion",
            "epsilon": 1e-2,
            "beta": 2.0,
            "alpha": 1.0,
            "delta_alpha": 1e-4,
        },
    )
    adasparse_ctx = adasparse_model({"scenario_features": scenario_tensor, "agnostic_features": agnostic_tensor})
    assert adasparse_ctx["logits"].shape == (batch_size, 1)
    assert adasparse_ctx["prediction"].shape == adasparse(original_batch).shape

    epnet = MultiDomainEPNet(
        sce_features=scenario_features,
        agn_features=agnostic_features,
        fcn_dims=[8],
    )
    epnet.eval()
    epnet_model = build_model_from_genome(
        GENOMES / "multi_domain_epnet_skillified.yaml",
        runtime_params={
            "scenario_vocab_sizes": [domain_num],
            "agnostic_vocab_sizes": [feature.vocab_size for feature in agnostic_features],
            "embedding_dim": embedding_dim,
            "scenario_dim": embedding_dim,
            "agnostic_dim": len(agnostic_features) * embedding_dim,
            "fcn_dims": [8],
        },
    )
    epnet_ctx = epnet_model({"scenario_features": scenario_tensor, "agnostic_features": agnostic_tensor})
    assert epnet_ctx["scenario_gate"].shape == (batch_size, len(agnostic_features) * embedding_dim)
    assert epnet_ctx["prediction"].shape == epnet(original_batch).shape

    id_features = _sparse_features([19], embedding_dim, prefix="id")
    ppnet_agnostic_features = _sparse_features([23, 29], embedding_dim, prefix="ppagn") + scenario_features
    ppnet_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in id_features + ppnet_agnostic_features if feature.name != "domain_indicator"}
    ppnet_batch["domain_indicator"] = domain_ids
    ppnet = MultiDomainPPNet(
        id_features=id_features,
        agn_features=ppnet_agnostic_features,
        domain_num=domain_num,
        fcn_dims=[8],
    )
    ppnet.eval()
    ppnet_model = build_model_from_genome(
        GENOMES / "multi_domain_ppnet_skillified.yaml",
        runtime_params={
            "id_vocab_sizes": [feature.vocab_size for feature in id_features],
            "agnostic_vocab_sizes": [feature.vocab_size for feature in ppnet_agnostic_features],
            "embedding_dim": embedding_dim,
            "input_dim": (len(id_features) + len(ppnet_agnostic_features)) * embedding_dim,
            "domain_num": domain_num,
            "fcn_dims": [8],
        },
    )
    ppnet_input = {
        "id_features": torch.stack([ppnet_batch[feature.name] for feature in id_features], dim=1),
        "agnostic_features": torch.stack([ppnet_batch[feature.name] for feature in ppnet_agnostic_features], dim=1),
        "domain_indicator": domain_ids,
    }
    ppnet_ctx = ppnet_model(ppnet_input)
    assert ppnet_ctx["domain_outputs"].shape == (batch_size, domain_num)
    assert ppnet_ctx["prediction"].shape == ppnet(ppnet_batch).shape


def test_multi_domain_star_sarnet_adaptdhm_genome_shapes_match_originals():
    torch.manual_seed(28)
    batch_size = 4
    domain_num = 3
    embedding_dim = 4
    features = _sparse_features([11, 13, 17], embedding_dim, prefix="md")
    input_dim = len(features) * embedding_dim
    domain_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_batch["domain_indicator"] = domain_ids
    sparse_batch = {
        "sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1),
        "domain_indicator": domain_ids,
    }

    star = MultiDomainStar(features=features, num_domains=domain_num, fcn_dims=[8], aux_dims=[8])
    star.eval()
    star_model = build_model_from_genome(
        GENOMES / "multi_domain_star_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "fcn_dims": [8],
            "aux_dims": [8],
        },
    )
    star_model.eval()
    star_ctx = star_model(sparse_batch)
    assert star_ctx["domain_outputs"].shape == (batch_size, domain_num)
    assert star_ctx["prediction"].shape == star(original_batch).shape

    sarnet = MultiDomainSarnet(
        features=features,
        domain_num=domain_num,
        domain_shared_expert_num=2,
        domain_specific_expert_num=1,
    )
    sarnet.eval()
    sarnet_model = build_model_from_genome(
        GENOMES / "multi_domain_sarnet_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "domain_shared_expert_num": 2,
            "domain_specific_expert_num": 1,
        },
    )
    sarnet_model.eval()
    sarnet_ctx = sarnet_model(sparse_batch)
    assert sarnet_ctx["prediction"].shape == sarnet(original_batch).shape

    adaptdhm = MultiDomainAdaptDHM(features=features, fcn_dims=[8], cluster_num=domain_num, beta=0.9, device="cpu")
    adaptdhm.eval()
    adaptdhm_model = build_model_from_genome(
        GENOMES / "multi_domain_adaptdhm_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "fcn_dims": [8],
            "cluster_num": domain_num,
            "beta": 0.9,
        },
    )
    adaptdhm_model.eval()
    adaptdhm_ctx = adaptdhm_model({"sparse_features": sparse_batch["sparse_features"]})
    assert adaptdhm_ctx["cluster_id"].shape == (batch_size,)
    assert adaptdhm_ctx["prediction"].shape == adaptdhm(original_batch).shape


def test_multi_domain_m2m_hamur_m3oe_genome_shapes_match_originals():
    torch.manual_seed(29)
    batch_size = 4
    domain_num = 3
    embedding_dim = 4
    features = _sparse_features([11, 13, 17], embedding_dim, prefix="advmd")
    domain_feature = [SparseFeature("domain_indicator", vocab_size=domain_num, embed_dim=embedding_dim)]
    m2m_features = features + domain_feature
    input_dim = len(features) * embedding_dim
    m2m_input_dim = len(m2m_features) * embedding_dim
    domain_ids = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in features}
    original_batch["domain_indicator"] = domain_ids
    sparse_batch = {
        "sparse_features": torch.stack([original_batch[feature.name] for feature in features], dim=1),
        "domain_indicator": domain_ids,
    }
    m2m_sparse_batch = {
        "sparse_features": torch.stack([original_batch[feature.name] for feature in m2m_features], dim=1),
        "domain_indicator": domain_ids,
    }

    transformer_dims = {"num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 16, "dropout": 0.0}
    m2m = MultiDomainM2M(
        features=m2m_features,
        domain_feature=domain_feature,
        domain_num=domain_num,
        num_experts=2,
        expert_output_size=8,
        transformer_dims=dict(transformer_dims),
    )
    m2m.eval()
    m2m_model = build_model_from_genome(
        GENOMES / "multi_domain_m2m_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in m2m_features],
            "embedding_dim": embedding_dim,
            "domain_vocab_sizes": [domain_num],
            "domain_embedding_dim": embedding_dim,
            "input_dim": m2m_input_dim,
            "domain_dim": embedding_dim,
            "num_experts": 2,
            "expert_output_size": 8,
            "transformer_dims": dict(transformer_dims),
        },
    )
    m2m_model.eval()
    m2m_ctx = m2m_model(m2m_sparse_batch)
    assert m2m_ctx["domain_context"].shape == (batch_size, embedding_dim)
    assert m2m_ctx["prediction"].shape == m2m(original_batch).shape

    hamur_small_dims = [8, 8]
    hamur_small = MultiDomainHamurSmall(
        features=features,
        domain_num=domain_num,
        fcn_dims=list(hamur_small_dims),
        hyper_dims=[6],
        k=4,
    )
    hamur_small.eval()
    hamur_small_model = build_model_from_genome(
        GENOMES / "multi_domain_hamur_small_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "fcn_dims": list(hamur_small_dims),
            "hyper_dims": [6],
            "k": 4,
        },
    )
    hamur_small_model.eval()
    hamur_small_ctx = hamur_small_model(sparse_batch)
    assert hamur_small_ctx["prediction"].shape == hamur_small(original_batch).shape

    hamur_large_dims = [8, 8, 8, 8, 8, 8, 8]
    hamur_large = MultiDomainHamurLarge(
        features=features,
        domain_num=domain_num,
        fcn_dims=list(hamur_large_dims),
        hyper_dims=[6],
        k=4,
    )
    hamur_large.eval()
    hamur_large_model = build_model_from_genome(
        GENOMES / "multi_domain_hamur_large_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "fcn_dims": list(hamur_large_dims),
            "hyper_dims": [6],
            "k": 4,
        },
    )
    hamur_large_model.eval()
    hamur_large_ctx = hamur_large_model(sparse_batch)
    assert hamur_large_ctx["prediction"].shape == hamur_large(original_batch).shape

    m3oe_dims = [8, 8, 8, 4]
    m3oe = MultiDomainM3oE(
        features=features,
        domain_num=domain_num,
        fcn_dims=list(m3oe_dims),
        expert_num=2,
        exp_d=0.5,
        exp_t=0.5,
        bal_d=0.5,
        bal_t=0.5,
        softmax_type=3,
        device="cpu",
    )
    m3oe.eval()
    m3oe_model = build_model_from_genome(
        GENOMES / "multi_domain_m3oe_skillified.yaml",
        runtime_params={
            "vocab_sizes": [feature.vocab_size for feature in features],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "domain_num": domain_num,
            "fcn_dims": list(m3oe_dims),
            "expert_num": 2,
            "exp_d": 0.5,
            "bal_d": 0.5,
        },
    )
    m3oe_model.eval()
    m3oe_ctx = m3oe_model(sparse_batch)
    assert m3oe_ctx["prediction"].shape == m3oe(original_batch).shape


def test_dssm_family_genome_shapes_match_originals():
    torch.manual_seed(16)
    batch_size = 4
    embedding_dim = 4
    user_features = _sparse_features([11, 13], embedding_dim, prefix="user")
    item_features = _sparse_features([17, 19], embedding_dim, prefix="item")
    hidden_dims = [8]
    original_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in user_features + item_features}

    dssm = DSSM(
        user_features=user_features,
        item_features=item_features,
        user_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
        item_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    dssm.eval()
    dssm_model = build_model_from_genome(
        GENOMES / "dssm_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "item_vocab_sizes": [feature.vocab_size for feature in item_features],
            "embedding_dim": embedding_dim,
            "user_input_dim": len(user_features) * embedding_dim,
            "item_input_dim": len(item_features) * embedding_dim,
            "user_hidden_dims": hidden_dims,
            "item_hidden_dims": hidden_dims,
            "temperature": 1.0,
        },
    )
    dssm_batch = {
        "user_features": torch.stack([original_batch[feature.name] for feature in user_features], dim=1),
        "item_features": torch.stack([original_batch[feature.name] for feature in item_features], dim=1),
    }
    dssm_ctx = dssm_model(dssm_batch)
    assert dssm_ctx["prediction"].shape == dssm(original_batch).shape

    pos_features = _sparse_features([17, 19], embedding_dim, prefix="pos")
    neg_features = _sparse_features([17, 19], embedding_dim, prefix="neg")
    fb_batch_original = {
        **{feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in user_features},
        **{feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in pos_features + neg_features},
    }
    facebook = FaceBookDSSM(
        user_features=user_features,
        pos_item_features=pos_features,
        neg_item_features=neg_features,
        user_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
        item_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    facebook.eval()
    fb_model = build_model_from_genome(
        GENOMES / "facebook_dssm_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "item_vocab_sizes": [feature.vocab_size for feature in pos_features],
            "embedding_dim": embedding_dim,
            "user_input_dim": len(user_features) * embedding_dim,
            "item_input_dim": len(pos_features) * embedding_dim,
            "user_hidden_dims": hidden_dims,
            "item_hidden_dims": hidden_dims,
            "temperature": 1.0,
        },
    )
    fb_batch = {
        "user_features": torch.stack([fb_batch_original[feature.name] for feature in user_features], dim=1),
        "pos_item_features": torch.stack([fb_batch_original[feature.name] for feature in pos_features], dim=1),
        "neg_item_features": torch.stack([fb_batch_original[feature.name] for feature in neg_features], dim=1),
    }
    fb_ctx = fb_model(fb_batch)
    fb_pos, fb_neg = facebook(fb_batch_original)
    assert fb_ctx["pos_logits"].shape == fb_pos.shape
    assert fb_ctx["neg_logits"].shape == fb_neg.shape

    senet = DSSMSENET(
        user_features=user_features,
        item_features=item_features,
        user_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
        item_params={"dims": hidden_dims, "dropout": 0.0, "activation": "relu"},
    )
    senet.eval()
    senet_model = build_model_from_genome(
        GENOMES / "dssm_senet_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "item_vocab_sizes": [feature.vocab_size for feature in item_features],
            "embedding_dim": embedding_dim,
            "num_user_fields": len(user_features),
            "num_item_fields": len(item_features),
            "reduction_ratio": 2,
            "user_input_dim": len(user_features) * embedding_dim,
            "item_input_dim": len(item_features) * embedding_dim,
            "user_hidden_dims": hidden_dims,
            "item_hidden_dims": hidden_dims,
            "temperature": 1.0,
        },
    )
    senet_ctx = senet_model(dssm_batch)
    assert senet_ctx["prediction"].shape == senet(original_batch).shape


def test_youtube_and_gru_matching_genome_shapes_match_originals():
    torch.manual_seed(17)
    batch_size = 4
    seq_len = 5
    n_neg = 3
    embedding_dim = 4
    user_features = _sparse_features([11, 13], embedding_dim, prefix="user")
    item = SparseFeature("item", vocab_size=23, embed_dim=embedding_dim, padding_idx=0)
    neg_item = SequenceFeature("neg_item", vocab_size=23, embed_dim=embedding_dim, pooling="concat", shared_with="item", padding_idx=0)
    history = SequenceFeature("history", vocab_size=23, embed_dim=embedding_dim, pooling="concat", shared_with="item", padding_idx=0)
    user_batch = {feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in user_features}
    sampled_batch = {
        **user_batch,
        "item": torch.randint(1, item.vocab_size, (batch_size,)),
        "neg_item": torch.randint(1, item.vocab_size, (batch_size, n_neg)),
        "history": torch.randint(1, item.vocab_size, (batch_size, seq_len)),
    }

    youtube = YoutubeDNN(
        user_features=user_features,
        item_features=[item],
        neg_item_feature=[neg_item],
        user_params={"dims": [embedding_dim], "dropout": 0.0, "activation": "relu"},
    )
    youtube.eval()
    youtube_model = build_model_from_genome(
        GENOMES / "youtube_dnn_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "embedding_dim": embedding_dim,
            "user_input_dim": len(user_features) * embedding_dim,
            "user_hidden_dims": [embedding_dim],
            "item_vocab_size": item.vocab_size,
            "temperature": 1.0,
        },
    )
    youtube_ctx = youtube_model(
        {
            "user_features": torch.stack([sampled_batch[feature.name] for feature in user_features], dim=1),
            "pos_item_features": sampled_batch["item"].unsqueeze(1),
            "neg_item_features": sampled_batch["neg_item"],
        }
    )
    assert youtube_ctx["scores"].shape == youtube(sampled_batch).shape

    sample_weight = DenseFeature("sample_weight")
    sbc_item_features = _sparse_features([17, 19], embedding_dim, prefix="sbc_item")
    sbc_original_batch = {
        **user_batch,
        **{feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in sbc_item_features},
        "sample_weight": torch.rand(batch_size) + 0.5,
    }
    sbc = YoutubeSBC(
        user_features=user_features,
        item_features=sbc_item_features,
        sample_weight_feature=[sample_weight],
        user_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        item_params={"dims": [8], "dropout": 0.0, "activation": "relu"},
        batch_size=batch_size,
        n_neg=2,
    )
    sbc.eval()
    sbc_model = build_model_from_genome(
        GENOMES / "youtube_sbc_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "item_vocab_sizes": [feature.vocab_size for feature in sbc_item_features],
            "embedding_dim": embedding_dim,
            "user_input_dim": len(user_features) * embedding_dim,
            "item_input_dim": len(sbc_item_features) * embedding_dim,
            "user_hidden_dims": [8],
            "item_hidden_dims": [8],
            "n_neg": 2,
            "temperature": 1.0,
        },
    )
    sbc_ctx = sbc_model(
        {
            "user_features": torch.stack([sbc_original_batch[feature.name] for feature in user_features], dim=1),
            "item_features": torch.stack([sbc_original_batch[feature.name] for feature in sbc_item_features], dim=1),
            "sample_weight": sbc_original_batch["sample_weight"],
        }
    )
    assert sbc_ctx["scores"].shape == sbc(sbc_original_batch).shape

    gru4rec = GRU4Rec(
        user_features=user_features,
        history_features=[history],
        item_features=[item],
        neg_item_feature=[neg_item],
        user_params={"dims": [embedding_dim], "dropout": 0.0, "activation": "relu"},
    )
    gru4rec.eval()
    gru_model = build_model_from_genome(
        GENOMES / "gru4rec_skillified.yaml",
        runtime_params={
            "user_vocab_sizes": [feature.vocab_size for feature in user_features],
            "embedding_dim": embedding_dim,
            "user_tower_input_dim": len(user_features) * embedding_dim + embedding_dim,
            "user_hidden_dims": [embedding_dim],
            "item_vocab_size": item.vocab_size,
            "temperature": 1.0,
        },
    )
    gru_ctx = gru_model(
        {
            "user_features": torch.stack([sampled_batch[feature.name] for feature in user_features], dim=1),
            "history_features": sampled_batch["history"],
            "pos_item_features": sampled_batch["item"].unsqueeze(1),
            "neg_item_features": sampled_batch["neg_item"],
        }
    )
    assert gru_ctx["scores"].shape == gru4rec(sampled_batch).shape


def test_session_matching_genome_shapes_match_originals():
    torch.manual_seed(18)
    batch_size = 4
    seq_len = 5
    embedding_dim = 4
    history = SequenceFeature("history", vocab_size=23, embed_dim=embedding_dim, pooling="concat", padding_idx=0)
    item = SparseFeature("item", vocab_size=23, embed_dim=embedding_dim, padding_idx=0)
    original_batch = {
        "history": torch.randint(1, history.vocab_size, (batch_size, seq_len)),
        "item": torch.randint(1, item.vocab_size, (batch_size,)),
    }

    narm = NARM(history, hidden_dim=embedding_dim, emb_dropout_p=0.0, session_rep_dropout_p=0.0, item_feature=item)
    narm.eval()
    narm_model = build_model_from_genome(
        GENOMES / "narm_skillified.yaml",
        runtime_params={"item_vocab_size": history.vocab_size, "embedding_dim": embedding_dim, "hidden_dim": embedding_dim},
    )
    narm_ctx = narm_model({"history_features": original_batch["history"], "item_features": original_batch["item"].unsqueeze(1)})
    assert narm_ctx["scores"].shape == narm(original_batch).shape

    stamp = STAMP(history, weight_std=0.01, emb_std=0.01, item_feature=item)
    stamp.eval()
    stamp_model = build_model_from_genome(
        GENOMES / "stamp_skillified.yaml",
        runtime_params={"item_vocab_size": history.vocab_size, "embedding_dim": embedding_dim},
    )
    stamp_ctx = stamp_model({"history_features": original_batch["history"], "item_features": original_batch["item"].unsqueeze(1)})
    assert stamp_ctx["scores"].shape == stamp(original_batch).shape


def test_multi_interest_and_sine_genome_shapes_match_originals():
    torch.manual_seed(19)
    batch_size = 4
    seq_len = 5
    n_neg = 3
    embedding_dim = 4
    interest_num = 3
    user_features = _sparse_features([11, 13], embedding_dim, prefix="mi_user")
    item = SparseFeature("item", vocab_size=23, embed_dim=embedding_dim, padding_idx=0)
    history = SequenceFeature("history", vocab_size=23, embed_dim=embedding_dim, pooling="concat", shared_with="item", padding_idx=0)
    neg_item = SequenceFeature("neg_item", vocab_size=23, embed_dim=embedding_dim, pooling="concat", shared_with="item", padding_idx=0)
    original_batch = {
        **{feature.name: torch.randint(0, feature.vocab_size, (batch_size,)) for feature in user_features},
        "history": torch.randint(1, item.vocab_size, (batch_size, seq_len)),
        "item": torch.randint(1, item.vocab_size, (batch_size,)),
        "neg_item": torch.randint(1, item.vocab_size, (batch_size, n_neg)),
    }
    genome_batch = {
        "user_features": torch.stack([original_batch[feature.name] for feature in user_features], dim=1),
        "history_features": original_batch["history"],
        "pos_item_features": original_batch["item"].unsqueeze(1),
        "neg_item_features": original_batch["neg_item"],
    }
    common_params = {
        "user_vocab_sizes": [feature.vocab_size for feature in user_features],
        "embedding_dim": embedding_dim,
        "user_dim": len(user_features) * embedding_dim,
        "item_vocab_size": item.vocab_size,
        "seq_len": seq_len,
        "interest_num": interest_num,
        "temperature": 1.0,
    }

    mind = MIND(user_features, [history], [item], [neg_item], max_length=seq_len, interest_num=interest_num)
    mind.eval()
    mind_ctx = build_model_from_genome(GENOMES / "mind_skillified.yaml", runtime_params=common_params)(genome_batch)
    assert mind_ctx["scores"].shape == mind(original_batch).shape

    comirec_sa = ComirecSA(user_features, [history], [item], [neg_item], interest_num=interest_num)
    comirec_sa.eval()
    sa_ctx = build_model_from_genome(GENOMES / "comirec_sa_skillified.yaml", runtime_params=common_params)(genome_batch)
    assert sa_ctx["scores"].shape == comirec_sa(original_batch).shape

    comirec_dr = ComirecDR(user_features, [history], [item], [neg_item], max_length=seq_len, interest_num=interest_num)
    comirec_dr.eval()
    dr_ctx = build_model_from_genome(GENOMES / "comirec_dr_skillified.yaml", runtime_params=common_params)(genome_batch)
    assert dr_ctx["scores"].shape == comirec_dr(original_batch).shape

    sine = SINE(
        history_features=["history"],
        item_features=["item"],
        neg_item_features=["neg_item"],
        num_items=item.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=6,
        num_concept=7,
        num_intention=interest_num,
        seq_max_len=seq_len,
        num_heads=1,
    )
    sine.eval()
    sine_ctx = build_model_from_genome(
        GENOMES / "sine_skillified.yaml",
        runtime_params={
            "item_vocab_size": item.vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": 6,
            "num_concept": 7,
            "num_intention": interest_num,
            "seq_len": seq_len,
            "num_heads": 1,
            "temperature": 1.0,
        },
    )(
        {
            "history_features": original_batch["history"],
            "pos_item_features": original_batch["item"].unsqueeze(1),
            "neg_item_features": original_batch["neg_item"],
        }
    )
    sine_original_batch = {
        "history": original_batch["history"],
        "item": original_batch["item"],
        "neg_item": original_batch["neg_item"].unsqueeze(1),
    }
    assert sine_ctx["scores"].shape == sine(sine_original_batch).shape


def test_generative_genome_shapes_are_skillified():
    torch.manual_seed(21)
    batch_size = 3
    seq_len = 4
    vocab_size = 23
    embedding_dim = 8

    common_sequence_params = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "max_seq_len": seq_len,
        "dqk": 4,
        "dv": 4,
    }
    seq_batch = torch.randint(1, vocab_size, (batch_size, seq_len))
    time_diffs = torch.arange(seq_len).float().unsqueeze(0).expand(batch_size, -1)

    hstu_model = build_model_from_genome(GENOMES / "hstu_skillified.yaml", runtime_params=common_sequence_params)
    hstu_ctx = hstu_model({"seq_features": seq_batch, "time_diffs": time_diffs})
    assert hstu_ctx["logits"].shape == (batch_size, seq_len, vocab_size)
    ok, message = validate_model_forward(
        hstu_model,
        {"seq_features": seq_batch, "time_diffs": time_diffs},
        required_outputs=["logits"],
    )
    assert ok, message

    transformer_params = {key: value for key, value in common_sequence_params.items() if key not in {"dqk", "dv"}}
    hllm_model = build_model_from_genome(GENOMES / "hllm_skillified.yaml", runtime_params=transformer_params)
    hllm_ctx = hllm_model({"seq_features": seq_batch})
    assert hllm_ctx["logits"].shape == (batch_size, seq_len, vocab_size)

    tiger_model = build_model_from_genome(GENOMES / "tiger_skillified.yaml", runtime_params=transformer_params)
    tiger_ctx = tiger_model({"seq_features": seq_batch})
    assert tiger_ctx["logits"].shape == (batch_size, seq_len, vocab_size)

    rqvae_model = build_model_from_genome(
        GENOMES / "rqvae_skillified.yaml",
        runtime_params={"input_dim": 6, "latent_dim": 4, "num_codes": 8},
    )
    dense_input = torch.randn(batch_size, 6)
    rqvae_ctx = rqvae_model({"dense_input": dense_input})
    assert rqvae_ctx["quantized"].shape == (batch_size, 4)
    assert rqvae_ctx["reconstruction"].shape == (batch_size, 6)
    assert rqvae_ctx["rqvae_loss"].shape == ()
