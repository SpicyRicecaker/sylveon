Get into gamedev WGPU raytracer (https://www.youtube.com/watch?v=Gv0EiQfDI7w)

WGPU web documentation (really mention that this is much better than opengl) https://www.w3.org/TR/webgpu/#gpusampler
WGSL web documentation (much better than opengl) https://www.w3.org/TR/WGSL/#type-sampler
WGPU rust documentation (https://docs.rs/wgpu/latest/wgpu/struct.Features.html)

Have no idea how to stream object/scene data into WGSL. But we need to do this if we're going to procedurally generate meshes. Also need to figure out how to do ray triangle intersections.

Good program to show displacements (https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html), Section of Webgpu Spec that says this: (https://www.w3.org/TR/WGSL/#alignment-and-size).