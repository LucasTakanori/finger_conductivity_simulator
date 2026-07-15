from finger_sim.mesh import discover_meshes, load_ring_mesh


def test_bundled_mesh_element_counts():
    paths = discover_meshes()
    assert len(paths) == 40
    assert {
        "b035_US060",
        "b035_US150",
        "b045_US060",
        "b045_US150",
        "subject006_US120",
        "subject001_US140_estimated",
    } <= set(paths)
    assert len(load_ring_mesh(paths["subject006_US120"]).elements) == 1330
    assert len(load_ring_mesh(paths["subject001_US140_estimated"]).elements) == 1524
