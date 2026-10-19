# Experimental global SfM mapper

`spirula sfm auto IMAGES/ -o WORKSPACE/ --mapper global` selects the global
initialization. `spirula sfm map` accepts the same option for an existing
feature and match database. The usual finishing, gauge and export stages follow.

The mapper bootstraps camera groups using the existing calibration machinery,
re-estimates relative poses from verified correspondences, and computes camera
orientations. Rotation averaging initializes from a spanning forest ranked by
triangle consistency and inlier count, then uses robust geodesic updates and
rejects inconsistent edges. Components are solved independently.

Accepted matches must also pass cheirality and a minimum pairwise parallax test.
They are joined into multi-view tracks. Nearby features from the same image are
reduced to the one supported by the most matches; more distant conflicts discard
the track. After positioning, observations inconsistent with the
estimated rays and tracks with insufficient triangulation angle are removed.
The angular threshold is 1 degree with a focal prior and 2 degrees without one,
matching the corresponding GLOMAP filtering stages.

The Vulkan positioning solver minimizes a robust camera-ray cost over camera
centres, points and a positive scale per observation. It uses LM and an implicit
Jacobi-preconditioned conjugate-gradient solve, with a fixed frame centre. An
observation scale is fixed when no calibrated rig baseline sets scale.
A Schur reduction eliminates observation scales;
the remaining system includes camera centres and points. Per-observation and
vector operations run on the GPU in fp32; camera contributions are reduced
cooperatively, while each point is accumulated by one thread. The host sums
partial reductions and controls the iterations. A
double-precision CPU reference is used in tests. The resulting `Reconstruction`
is passed to the existing bundle adjustment, assembly, recovery and export.
Each initial refinement round runs BA with frame rotations fixed, followed by
joint BA; both stages retain the available pose priors.

This is an experimental implementation, not a port or a validated equivalent
of GLOMAP. Its current limits matter when choosing a test capture:

- Positioning has not been profiled on large scenes or validated on a physical
  GPU. The reduced system still includes points, and fp32 arithmetic and host
  synchronization may limit precision and throughput.
- Unknown focal lengths use the existing camera bootstrap, not GLOMAP's global
  calibration. Pairwise planar and panoramic geometries are skipped.
- Wrong bridges without cycles cannot be detected by rotation cycle checks;
  repeated structure and long, weakly connected sequences need real-data tests.
- A track with distant conflicts is discarded rather than split, and track union arrays
  are sized for all detected features. This can lose valid observations and use
  substantial memory.
- Fixed, supplied rig extrinsics constrain rotation averaging and positioning;
  known baselines set scale in the supplied extrinsics' units. Refinable rig extrinsics
  and GPS/IMU priors do not constrain these global stages. A synthetic fixed
  rig is covered through positioning, BA and export; fisheye, equirectangular
  cameras and real rigs have not been validated end to end.

The `sfm_global_*`, `sfm_rotation_average_test` and
`sfm_relative_view_graph_test` tests exercise the stages with synthetic data,
including outliers and disconnected graphs. Synthetic success is not evidence
of parity with GLOMAP or of quality on real image collections.
`sfm_fixed_rotation_ba_test`, also with `--vulkan`, checks dense and CG BA
with pose priors, fixed rotations and a subsequent free solve.
