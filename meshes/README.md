# Bundled PVI ring meshes

The repository includes every ring mesh from both authoritative PVI MATLAB
collections so the desktop app does not depend on an HPC filesystem.

## Included bundles

- `collections/b035/US060` through `collections/b035/US150`: all 19 b035
  variants in 5-unit steps.
- `collections/b045/US060` through `collections/b045/US150`: all 19 b045
  variants in 5-unit steps.
- `subject006_US120`: US120 inverse mesh (1,330 elements), forward mesh
  (5,320 elements), 40×40 mappings, and manifest. US120 is confirmed by the
  historical subject-006 image mask.
- `subject001_US140_estimated`: US140 inverse mesh (1,524 elements), forward
  mesh (6,096 elements), 40×40 mappings, and manifest. This selection is an
  estimate; the candidate residuals were close and the manifest warning must
  remain visible.

Each collection bundle contains an inverse mesh, forward mesh, 40×40 mapping,
and provenance manifest. The b035 collection already contains 40×40 image
mappings; b045 mappings are rebuilt at 40×40 by the same exporter used for the
subject-selected bundles.

The HDF5 files are exported derivatives of `mesh_collection_b035.mat` and
`mesh_collection_b045.mat` from the Peripheral-Vascular-Impedance-Imaging
project. Do not rename files independently of their manifest.
