def test_viz_module_imports_and_exposes_overlay():
    from glider.vision.pose import viz

    # The public overlay entry point exists and is callable.
    assert callable(viz.overlay_frames)
