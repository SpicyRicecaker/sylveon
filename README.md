Get into gamedev WGPU raytracer (https://www.youtube.com/watch?v=Gv0EiQfDI7w)

WGPU web documentation (really mention that this is much better than opengl) https://www.w3.org/TR/webgpu/#gpusampler
WGSL web documentation (much better than opengl) https://www.w3.org/TR/WGSL/#type-sampler
WGPU rust documentation (https://docs.rs/wgpu/latest/wgpu/struct.Features.html)

Have no idea how to stream object/scene data into WGSL. But we need to do this if we're going to procedurally generate meshes. Also need to figure out how to do ray triangle intersections.

Good program to show displacements (https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html), Section of Webgpu Spec that says this: (https://www.w3.org/TR/WGSL/#alignment-and-size).

Random GPU algorithms (https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39)

[Chris Biscardi](https://www.youtube.com/watch?v=vblsZgBcgyw) gave the suggestion for me to use [Renderdoc](https://renderdoc.org/). Turns out that the color information is stored in  `R8G8B8A8_UNORM` format while it is in the storage buffer. This storage buffer image looks exactly like the one in WebGPU. However, when the texture gets read by the fragment shader and returned, each color is then converted into the `B8G8R8A8_SRGB` format. [This Excellent Post](https://community.khronos.org/t/noob-difference-between-unorm-and-srgb/106132/7) talks about color spaces, and what the screen expects.
