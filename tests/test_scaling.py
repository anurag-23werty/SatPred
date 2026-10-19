import numpy as np
import pytest

from sathealth.features.scaling import ChannelScaler, ScalerParams


@pytest.fixture
def data() -> np.ndarray:
    rng = np.random.default_rng(1)
    value = rng.normal(loc=5.0, scale=2.0, size=20_000)
    commands = (rng.random((20_000, 3)) > 0.9).astype(float)
    return np.column_stack([value, commands])


@pytest.mark.parametrize("kind", ["robust", "standard"])
def test_scaled_values_are_unit_sigma(data: np.ndarray, kind: str) -> None:
    out = ChannelScaler(kind).fit_transform(data)  # type: ignore[arg-type]
    assert abs(np.median(out[:, 0])) < 0.05
    assert abs(out[:, 0].std() - 1.0) < 0.05, "robust scale is calibrated to σ for Gaussians"


def test_command_columns_pass_through(data: np.ndarray) -> None:
    out = ChannelScaler("robust").fit_transform(data)
    np.testing.assert_array_equal(out[:, 1:], data[:, 1:])


def test_robust_ignores_outliers_in_train(data: np.ndarray) -> None:
    contaminated = data.copy()
    contaminated[:200, 0] = 1e4  # 1% unlabelled anomalies hiding in "nominal" train
    robust = ChannelScaler("robust").fit(contaminated).params.scale[0]
    standard = ChannelScaler("standard").fit(contaminated).params.scale[0]
    assert robust == pytest.approx(2.0, rel=0.05)
    assert standard > 100


def test_constant_channel_does_not_explode() -> None:
    x = np.column_stack([np.full(500, -1.0), np.zeros(500)])
    scaler = ChannelScaler("robust").fit(x)
    assert scaler.params.degenerate == (True,)
    assert scaler.params.scale == (1.0,)
    test = np.column_stack([np.array([-1.0, 258.1]), np.zeros(2)])  # like channel M-6
    np.testing.assert_allclose(scaler.transform(test)[:, 0], [0.0, 259.1])


def test_mostly_flat_channel_falls_back_to_std() -> None:
    value = np.zeros(1000)
    value[::10] = 1.0  # IQR is 0, but the signal does move
    scaler = ChannelScaler("robust").fit(value[:, None])
    assert scaler.params.scale[0] == pytest.approx(value.std())
    assert scaler.params.degenerate == (False,)


def test_inverse_roundtrip_and_serialisation(data: np.ndarray) -> None:
    scaler = ChannelScaler("robust").fit(data)
    restored = ChannelScaler.from_params(ScalerParams.from_dict(scaler.params.to_dict()))
    np.testing.assert_allclose(restored.inverse_transform(restored.transform(data)), data)


def test_transform_does_not_mutate_input(data: np.ndarray) -> None:
    original = data.copy()
    ChannelScaler("robust").fit_transform(data)
    np.testing.assert_array_equal(data, original)


def test_none_is_identity(data: np.ndarray) -> None:
    np.testing.assert_array_equal(ChannelScaler("none").fit_transform(data), data)


def test_unfitted_scaler_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ChannelScaler().transform(np.zeros((2, 1)))
