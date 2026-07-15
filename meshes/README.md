# Bundled PVI ring meshes

The repository includes the currently exported ring mesh bundles from the
2D_GCNM workspace so the desktop app does not depend on an HPC filesystem.

## Included bundles

- `subject006_US120`: US120 inverse mesh (1,330 elements), forward mesh
  (5,320 elements), 40×40 mappings, and manifest. US120 is confirmed by the
  historical subject-006 image mask.
- `subject001_US140_estimated`: US140 inverse mesh (1,524 elements), forward
  mesh (6,096 elements), 40×40 mappings, and manifest. This selection is an
  estimate; the candidate residuals were close and the manifest warning must
  remain visible.

The HDF5 files are exported derivatives of
`mesh_collection_b045.mat` from the Peripheral-Vascular-Impedance-Imaging
project. Do not rename files independently of their manifest.

