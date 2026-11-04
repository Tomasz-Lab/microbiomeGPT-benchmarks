import pytest
import torch


@torch.no_grad()
def test_output_shapes(small_model, small_model_inputs, small_model_config, small_training_config):
    # Given
    batch_size = small_training_config["batch_size"]
    pathways, taxonomy = small_model_inputs

    # When
    taxonomy_pred, pathways_pred, latent = small_model(pathways, taxonomy)

    # Then
    assert taxonomy_pred.shape == (batch_size, small_model_config["num_bacteria"], 1)
    assert pathways_pred.shape == (batch_size, small_model_config["num_pathways"], 1)
    assert latent.shape == (batch_size, small_model_config["latent_size"])


# TODO Remove or modify while implementing #17 Extract Unified Data Pipeline (BacteriaDataModule)
@torch.no_grad()
def test_takes_pathways_first(small_model, small_model_inputs):
    # Given
    pathways, taxonomy = small_model_inputs

    # When / Then
    with pytest.raises(IndexError):
        small_model(taxonomy, pathways)


@torch.no_grad()
def test_samples_are_independent_in_eval(small_model, small_model_inputs, small_training_config):
    # Given
    batch_size = small_training_config["batch_size"]
    pathways, taxonomy = small_model_inputs

    # When
    batch_outputs = small_model(pathways, taxonomy)
    single_outputs = [
        small_model(pathways[i : i + 1], taxonomy[i : i + 1]) for i in range(batch_size)
    ]

    # Then
    for i, sample_outputs in enumerate(single_outputs):
        for batch_out, single_out in zip(batch_outputs, sample_outputs, strict=True):
            torch.testing.assert_close(batch_out[i : i + 1], single_out)


@torch.no_grad()
def test_same_seed_gives_same_outputs(small_model, small_model_same_seed, small_model_inputs):
    # Given
    model_a = small_model
    model_b = small_model_same_seed
    pathways, taxonomy = small_model_inputs

    # When
    outputs_a = model_a(pathways, taxonomy)
    outputs_b = model_b(pathways, taxonomy)

    # Then
    for out_a, out_b in zip(outputs_a, outputs_b, strict=True):
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


@torch.no_grad()
def test_checkpoint_round_trip(small_model, small_model_other_seed, small_model_inputs, tmp_path):
    # Given
    pathways, taxonomy = small_model_inputs
    expected = small_model(pathways, taxonomy)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(small_model.state_dict(), checkpoint_path)
    fresh_model = small_model_other_seed
    assert not torch.equal(fresh_model(pathways, taxonomy)[2], expected[2])

    # When
    fresh_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    got = fresh_model(pathways, taxonomy)

    # Then
    for exp_out, got_out in zip(expected, got, strict=True):
        torch.testing.assert_close(got_out, exp_out, rtol=0, atol=0)
