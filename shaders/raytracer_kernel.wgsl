@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;

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

@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var<uniform> window_size: vec2u;

struct Material {
    color: vec3f,
    ambient: vec3f,
}

struct Sphere {
    center: vec3f,
    radius: f32
}

struct HitRecord {
    p: vec3f,
    t: f32,
    normal: vec3f,
    hit: bool
}

// http://www.jcgt.org/published/0009/03/02/
fn pcg3d(p: vec3u) -> vec3u {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

fn rand33(f: vec3f) -> vec3f { 
    return vec3f(pcg3d(bitcast<vec3u>(f))) / f32(0xffffffffu); 
}

fn random_in_unit_sphere(seed: vec3f) -> vec3f {
    var out: vec3f;
    while (true) {
        let r = rand33(seed);
        if (pow(length(r), 2.) < 1.) {
            out = r;
            break;
        }
    }
    return out;
}

fn random_in_hemisphere(ray_incident: vec3f, normal: vec3f) -> vec3f {
    var ray_reflected: vec3f;
    if dot(ray_incident, normal) < 0. {
        ray_reflected = -ray_incident;
    } else {
        ray_reflected = ray_incident;
    }
    return ray_reflected;
}

struct RangeInclusive {
    min: f32,
    max: f32
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct StackFrame {
    outgoing: Ray,
    depth: u32,
}

fn is_in_range(r: RangeInclusive, n: f32) -> bool {
    return r.min <= n && n <= r.max;
}

var<private> spheres: array<Sphere, 2>;
var<private> global_invocation_id: vec2<i32>;

fn world_get_background_color() -> vec3f {
    let blue = vec3f(135., 206., 235.) / 256.;
    let white = vec3f(1., 1., 1.);
    // lerp based on y position
    let color_ratio = f32(global_invocation_id.y) / f32(window_size.y);
    let pixel_color = (1. - color_ratio) * blue + color_ratio * white;
    return pixel_color;
}

fn world_get_normal(p: vec3f, id: i32) -> vec3f {
    return (-spheres[id].center + p) / spheres[id].radius;
}

struct Intersect {
    t: f32,
    id: i32
}

fn world_get_intersect(ray: Ray) -> Intersect {
    var t_range = RangeInclusive(0.001, 1000.);
    var i: i32 = 0;

    var intersect = Intersect();
    intersect.t = 0.;
    intersect.id = -1;

    loop {
        if i >= 2 { break; }

        let c_o = -spheres[i].center + ray.origin;
        let a = dot(ray.direction, ray.direction);
        let b = 2. * dot(ray.direction, c_o);
        let c = dot(c_o, c_o) - pow(spheres[i].radius, 2.);

        let det = pow(b, 2.) - 4. * a * c;

        if det < 0. {
            continue;
        }

        // why not always take smallest t?
        var t = (-b - sqrt(det)) / (2. * a);

        // For this specific ray, do not accept any hits farther away
        // from the current hit
        if !is_in_range(t_range, t) {
            // check with the positive t
            t = (-b + sqrt(det)) / (2. * a);
            if !is_in_range(t_range, t) {
                continue;
            }
        }

        intersect.t = t;
        intersect.id = i;

        t_range.max = min(t_range.max, t);

        continuing {
            i += 1;
        }
    }

    return intersect;
}

struct Light {
    center: vec3f
}

var<private> sun: Light;

// if direct view of the sun
fn world_get_direct_light_at_point(p: vec3f) -> bool {
    var ray = Ray();
    ray.origin = p;
    ray.direction = normalize(-p + sun.center);

    let intersect = world_get_intersect(ray);

    var out = true;
    if (intersect.id >= 0) {
        // find the time it takes for sun's ray to reach point
        // let t_to_p = length(-p + sun.center) / length(ray.direction);

        // sun's ray should reach point faster than (or in equal time of) the shortest intersection point
        // i.e., t_to_p <= t_to_intersect -> t_to_p - t_to_intersect <= 0
        // generous margin of error
        // if !(abs(t_to_p - intersect.t) <= 0.0001) {
        out = false;
        // }
    } 
    return out;
}

fn world_get_color(ray: Ray) -> vec3f {
    // at bottom, completely white
    // at top, sky blue

    // we can't directly take this value for the pixel yet, since our path tracer is going to add that in for us
    var pixel_color = vec3f(0., 0., 0.);

    let max_depth = 1;
    var depth = 0;

    loop {
        if (depth == max_depth) { break; }

        let intersect = world_get_intersect(ray);
        
        // assume sky is just empty
        if intersect.id < 0 {
            pixel_color = world_get_background_color();
            break;
        }

        let p = ray.origin + intersect.t * ray.direction;

        if world_get_direct_light_at_point(p) {
            let normal = world_get_normal(p, intersect.id);

            // make sure this is a color between 0 and 1 lol
            pixel_color = .5 * (normal + vec3(1., 1., 1.));
        }

        // // get direct lighting
        // var direct: vec3f = world_get_direct_light_at_point(p);

        // let new_direction = random_in_hemisphere(ray.direction, normal);

        // pixel_color = BRDF(ray.direction, new_direction) * ray.normal + direct;

        // ray.origin = p;
        // ray.direction = new_direction;

        continuing {
            depth += 1;
        }
    }

    return pixel_color;
}

// number of pixels handled by this function
@compute @workgroup_size(1,1,1)
// globalinvocationid: coordinate of current pixel
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_pos: vec2<i32> = vec2(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));
    global_invocation_id = screen_pos;

    // initialize spheres
    spheres[0] = Sphere(vec3(0., -100.5, -1.), 100.);
    spheres[1] = Sphere(vec3(0., 0., -1.), .5);
    // sun
    sun.center = vec3(0., 100., 0.);

    // screen space is [-1., 1.]
    let height = 2. * tan(cam.fov_y) * length(cam.focal_length);
    let width = height * cam.aspect_ratio;

    let viewport_u = vec3f(width, 0., 0.);
    let viewport_v = vec3f(0., -height, 0.);

    // find the top left node
    // treat this screen as if it only exists in two dimensions
    let du = viewport_u / f32(window_size.x);
    let dv = viewport_v / f32(window_size.y);
    let tl_pixel = cam.eye + cam.focal_length * cam.direction - .5 * (viewport_u + viewport_v) + .5 * (du + dv);

    let pixel = tl_pixel + du * f32(GlobalInvocationID.x) + dv * f32(GlobalInvocationID.y);

    // ray!
    // make sure to center it at the current eye position
    var ray = Ray();
    // this works, but only when we don't rotate the camera at all
    ray.origin = cam.eye;
    // by default should be skybox color
    ray.direction = -cam.eye + pixel;

    // check for intersection of ray with thing
    var pixel_color = world_get_color(ray);

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.));
}
