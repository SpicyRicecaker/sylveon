@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<uniform> window_size: vec2u;

struct Camera {
    eye: vec3f,
    focal_length: f32,
    direction: vec3f,
    aspect_ratio: f32,
    normal: vec3f,
    fov_y: f32,
    right: vec3f,
    dummy1: f32
}

struct VertexInput {
    @location(0) coord: vec3f
}

struct VertexOutput {
    @builtin(position) position: vec4f
}

@vertex
fn vert_main(vertex_input: VertexInput) -> VertexOutput {
    // build projection matrix
    // build view matrix
    var vertex_output: VertexOutput;
    vertex_output.position = vec4f(vertex_input.coord, 1.);
    
    return vertex_output;
}

@fragment
fn frag_main(vertex_ouput: VertexOutput) -> @location(0) vec4f {
    return vec4f(1., 1., 1., 1.);
}