# Cluster-fit reference sources

`cpu/` contains the complete BC1–BC5 codec sources and shared cluster-fit
algorithm from the cluster-fit implementation. `shaders/` contains its
BC1/BC2 GPU shaders and their binomial, reduction, and fitting dependencies.
These files are frozen references, independent of the active encoders.

The active CPU codecs use PA + up to four LS passes + one discrete neighbor
round. The wgpu and mev routes use the shared `pa_ls_ns.wgsl` implementation.
BC6H has a separate HDR encoder and is outside this archive.

The public `jkl::algos::cluster_fit` module is retained inside the `jkl`
package for source compatibility; production BC1–BC5 encoders do not call it.
The GPU experiment references in `gpu_search.rs` and `format_color.rs` load
the frozen shaders here. Saved experiment reports retain the measurements
made with their documented versions. A new experiment's CPU production
reference always calls the active CPU encoder from its checkout.

CPU reference limitations: BC2 and BC4 contain their original valid-texel
index-packing defects. BC3 and BC5 inherit the BC4 behavior. The GPU search
has its original binomial iteration bounds. These sources preserve those
semantics for inspection, rather than defining the active codec behavior.
