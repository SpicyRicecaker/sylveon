
![Screenshot 2024-09-30 135821](https://github.com/user-attachments/assets/562dfbea-b0e6-4e42-b24a-94c357c8be1b)
![Screenshot 2024-09-30 135801](https://github.com/user-attachments/assets/6c082a44-8f1e-4fe5-b4f8-e056e08a68ad)


Get into gamedev WGPU raytracer (https://www.youtube.com/watch?v=Gv0EiQfDI7w)

WGPU web documentation (really mention that this is much better than opengl) https://www.w3.org/TR/webgpu/#gpusampler
WGSL web documentation (much better than opengl) https://www.w3.org/TR/WGSL/#type-sampler
WGPU rust documentation (https://docs.rs/wgpu/latest/wgpu/struct.Features.html)

Have no idea how to stream object/scene data into WGSL. But we need to do this if we're going to procedurally generate meshes. Also need to figure out how to do ray triangle intersections.

Good program to show displacements (https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html), Section of Webgpu Spec that says this: (https://www.w3.org/TR/WGSL/#alignment-and-size).

Random GPU algorithms (https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39)

[Chris Biscardi](https://www.youtube.com/watch?v=vblsZgBcgyw) gave the suggestion for me to use [Renderdoc](https://renderdoc.org/). Turns out that the color information is stored in  `R8G8B8A8_UNORM` format while it is in the storage buffer. This storage buffer image looks exactly like the one in WebGPU. However, when the texture gets read by the fragment shader and returned, each color is then converted into the `B8G8R8A8_SRGB` format. [This Excellent Post](https://community.khronos.org/t/noob-difference-between-unorm-and-srgb/106132/7) talks about color spaces, and what the screen expects.

(https://www.shadertoy.com/view/tl23Rm) Shadertoy shader primititve (https://reindernijhoff.net/2019/06/ray-tracing-primitives/)

(https://iquilezles.org/articles/simplepathtracing/) articesl on gpu pathtracing

(https://en.wikipedia.org/wiki/Path_tracing) highkey the wikipedia article on path tracing is goated. (https://en.wikipedia.org/wiki/Ray_casting) Ray casting -> (https://en.wikipedia.org/wiki/Ray_tracing_(graphics)#Recursive_ray_tracing_algorithm) Recursive Ray Tracing -> (https://en.wikipedia.org/wiki/Distributed_ray_tracing) Distribution Ray tracing -> Path Tracing. 

(https://news.ycombinator.com/item?id=34012094) Big hackernews post. Great comments.

(https://www.techspot.com/article/2485-path-tracing-vs-ray-tracing/) Path Tracing vs. Raytracer diff.

(https://www.techspot.com/article/1888-how-to-3d-rendering-rasterization-ray-tracing/) BVH Overview 

https://www.scratchapixel.com/ Raytracing in One Weekend but explained differently pog?

(https://www.scratchapixel.com/lessons/3d-basic-rendering/rendering-3d-scene-overview/introduction-light-transport.html) Unidirectional path tracing

(https://www.realtimerendering.com/raytracing.html) Includes so many websites listed above.

(https://www.youtube.com/watch?v=4gXPVoippTs&list=PLujxSBD-JXgnGmsn7gEyN28P1DnRZG7qi&index=3) Radiance model of simulation. BRDF, BTDF. BSDF. Physics of transmittance vs. absorption of light, intensity.
 
(https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html, https://www.youtube.com/watch?v=ZhN5-o397QI&list=PLujxSBD-JXgnGmsn7gEyN28P1DnRZG7qi&index=10, https://pbr-book.org/4ed/Cameras_and_Film/Camera_Interface, https://raytracing.github.io/books/RayTracingInOneWeekend.html) Trying to understand wtf perspective camera (w/ fov) means!!!

<!-- (https://www.youtube.com/watch?v=zc8b2Jo7mno) Gimbal lock in attempting to make 3D minecraft camera! (Blender Manipulator tool - rotate around the z-axis and then rotate around the x-axis) -->
Really weird camera rotations if you do yaw before pitch, since you're rotating from the x-axis and y-axis. Instead, pitch first before yaw allows you to rotate from the start in both ways.  (https://forum.gamemaker.io/index.php?threads/first-person-camera-rotation.38106/post-235130)

(https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/3d-meshes/transformation-matrices.html) interesting resource for WebGPU matrices. Basically call `transpose` right after passing in vectors into matrices and it makes much more sense.

(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4930030/) Definition of BSDF as a function which gives incident irradiance / scattered radiance.

(https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) iterative recursion

(https://www.shadertoy.com/view/7tBXDh) Clutch rt in one weekend shader implementation.

(https://www.reddit.com/r/rust_gamedev/comments/rozcgr/wgsl_alignment_workarounds/)
