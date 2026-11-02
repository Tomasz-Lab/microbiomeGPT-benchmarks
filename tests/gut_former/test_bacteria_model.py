import pytest
import torch


@torch.no_grad()
def test_output_shapes(model, inputs, dims):
    # Given
    batch_size = inputs.pathways.shape[0]

    # When
    taxonomy_pred, pathways_pred, latent = model(inputs.pathways, inputs.taxonomy)

    # Then
    assert taxonomy_pred.shape == (batch_size, dims.num_bacteria, 1)
    assert pathways_pred.shape == (batch_size, dims.num_pathways, 1)
    assert latent.shape == (batch_size, dims.latent_size)


@torch.no_grad()
def test_takes_pathways_first(model, inputs):
    # Given
    swapped_args = (inputs.taxonomy, inputs.pathways)

    # When / Then
    with pytest.raises(IndexError):
        model(*swapped_args)


@torch.no_grad()
def test_samples_are_independent_in_eval(model, inputs):
    # Given
    batch_size = inputs.pathways.shape[0]

    # When
    batch_outputs = model(inputs.pathways, inputs.taxonomy)
    single_outputs = [
        model(inputs.pathways[i : i + 1], inputs.taxonomy[i : i + 1]) for i in range(batch_size)
    ]

    # Then
    for i, sample_outputs in enumerate(single_outputs):
        for batch_out, single_out in zip(batch_outputs, sample_outputs, strict=True):
            torch.testing.assert_close(batch_out[i : i + 1], single_out)


@torch.no_grad()
def test_same_seed_gives_same_outputs(model_factory, inputs):
    # Given
    model_a = model_factory(seed=0).eval()
    model_b = model_factory(seed=0).eval()

    # When
    outputs_a = model_a(inputs.pathways, inputs.taxonomy)
    outputs_b = model_b(inputs.pathways, inputs.taxonomy)

    # Then
    for out_a, out_b in zip(outputs_a, outputs_b, strict=True):
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


@torch.no_grad()
def test_checkpoint_round_trip(model, model_factory, inputs, tmp_path):
    # Given
    expected = model(inputs.pathways, inputs.taxonomy)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    fresh_model = model_factory(seed=1).eval()
    assert not torch.equal(fresh_model(inputs.pathways, inputs.taxonomy)[2], expected[2])

    # When
    fresh_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    got = fresh_model(inputs.pathways, inputs.taxonomy)

    # Then
    for exp_out, got_out in zip(expected, got, strict=True):
        torch.testing.assert_close(got_out, exp_out, rtol=0, atol=0)
