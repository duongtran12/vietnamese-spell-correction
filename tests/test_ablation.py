from scripts.run_ablation import default_ablation_specs, plot_ablation, plot_error_types


def test_default_ablation_specs_cover_required_components() -> None:
    specs = {spec.name: spec for spec in default_ablation_specs()}

    assert specs["edit_frequency"].context_weight == 0.0
    assert specs["context_edit"].frequency_weight == 0.0
    assert specs["full_interpolated"].lambdas == (0.1, 0.3, 0.6)
    assert specs["unigram_context"].lambdas == (1.0, 0.0, 0.0)
    assert specs["bigram_context"].lambdas == (0.0, 1.0, 0.0)
    assert specs["trigram_context"].lambdas == (0.0, 0.0, 1.0)


def test_ablation_plot_helpers_write_nonempty_png_files(tmp_path) -> None:
    results = [
        {"label": "Baseline", "accuracy_at_1": 0.5, "accuracy_at_3": 0.7},
        {"label": "Full", "accuracy_at_1": 0.6, "accuracy_at_3": 0.8},
    ]
    ablation_path = tmp_path / "ablation.png"
    error_path = tmp_path / "errors.png"

    plot_ablation(results, ablation_path)
    plot_error_types({"delete": 0.5, "insert": 0.75}, error_path)

    assert ablation_path.stat().st_size > 0
    assert error_path.stat().st_size > 0
