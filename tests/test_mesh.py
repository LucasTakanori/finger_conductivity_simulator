from finger_sim.mesh import discover_meshes, load_ring_mesh


def test_bundled_mesh_element_counts():
    paths = discover_meshes()
    assert {"subject006_US120", "subject001_US140_estimated"} <= set(paths)
    assert len(load_ring_mesh(paths["subject006_US120"]).elements) == 1330
    assert len(load_ring_mesh(paths["subject001_US140_estimated"]).elements) == 1524

