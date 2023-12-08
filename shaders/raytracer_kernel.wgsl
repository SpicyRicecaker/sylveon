@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;

struct Sphere {
    center: vec3<f32>,
    radius: f32
}

struct Camera {
    eye: vec3<f32>,
    direction: vec3<f32>,
    normal: vec3<f32>,
    right: vec3<f32>,
    focal_length: f32
}

// number of pixels handled by this function
@compute @workgroup_size(1,1,1)
// globalinvocationid: coordinate of current pixel
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_pos: vec2<i32> = vec2(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    let sphere = Sphere(vec3(0., 0., -3.), .5);
    // check for intersection of ray with thing
    var camera: Camera;
    camera.eye = vec3(0., 0., 0.);
    camera.direction = vec3(0., 0., -1.);
    camera.normal = vec3(0., 1., 0.);
    camera.right = vec3(1., 0., 0.);
    camera.focal_length = 1.;
    // should be uniforms tbh
    let aspect_ratio: f32 = 16. / 9.;
    let width = 1.;
    let height = width / aspect_ratio;
    let MAX_WIDTH: u32 = u32(400);
    let MAX_HEIGHT: u32 = u32(f32(MAX_WIDTH) / aspect_ratio);
    // find the top left node
    let tl_pixel_corner: vec2<f32> = vec2(-width / 2., height / 2.);
    let du: vec2<f32> = vec2(width / f32(MAX_WIDTH), 0.);
    let dv: vec2<f32> = vec2(0., -height / f32(MAX_HEIGHT));
    let tl_pixel = tl_pixel_corner + du / 2. + dv / 2.;

    let pixel = tl_pixel + du * f32(GlobalInvocationID.x) + dv * f32(GlobalInvocationID.y);

    // ray!
    let origin = camera.eye;
    let direction = -camera.eye + vec3<f32>(pixel.xy, 0.) + camera.focal_length * camera.direction;

    let a = dot(direction, direction);
    let c_o = -sphere.center + origin;
    let b = 2. * dot(direction, c_o);
    let c = dot(c_o, c_o) - pow(sphere.radius, 2.);

    let det = pow(b, 2.) - 4. * a * c;

    var pixel_color = vec3(0., 0., 0.);

    if (det >= 0.) {
        // why not always take smallest t?
        let t = (-b - sqrt(det)) / 2. * a;
        let p = origin + t * direction;

        let normal = normalize(-sphere.center + p);
        // make sure this is a color between 0 and 1 lol

        pixel_color = (normal + vec3(1., 1., 1.)) / 2.;
    } 

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.));
}
