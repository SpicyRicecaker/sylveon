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

struct Material {
    albedo: vec3f,
    padding_1: f32,
    ambient: vec3f,
    padding_2: f32,
}

struct Triangle {
    points: array<vec4f, 3>,
    material: Material
}

struct Sphere {
    center: vec3f,
    radius: f32,
    material: Material,
}

@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var<uniform> window_size: vec2u;
@group(0) @binding(3) var<storage, read> triangles: array<Triangle>;

struct HitRecord {
    t: f32,
    p: vec3f,
    normal: vec3f,
    hit: bool,
    material: Material
}

// the following hashes were inspired by 
// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
// keep in mind each function only generates random numbers in the range
// of [0.,1.]

// https://www.pcg-random.org/
fn pcg(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn pcg2d(p: vec2u) -> vec2u {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    v.x += v.y * 1664525u; v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    return v;
}

// http://www.jcgt.org/published/0009/03/02/
fn pcg3d(p: vec3u) -> vec3u {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

fn rand11(f: f32) -> f32 { return f32(pcg(bitcast<u32>(f))) / f32(0xffffffffu); }

fn rand22(f: vec2f) -> vec2f { return vec2f(pcg2d(bitcast<vec2u>(f))) / f32(0xffffffffu); }

fn rand33(f: vec3f) -> vec3f { return vec3f(pcg3d(bitcast<vec3u>(f))) / f32(0xffffffffu); }

fn random_in_unit_sphere(seed: vec3f) -> vec3f {
    var out: vec3f;
    loop {
        let r = 2. * rand33(seed) - vec3(1., 1., 1.);
        if (pow(length(r), 2.) < 1.) {
            out = r;
            break;
        }
    }
    return out;
}

fn random_unit_vec(seed: vec3f) -> vec3f {
    return normalize(random_in_unit_sphere(seed));
}

fn random_in_hemisphere(ray_incident: vec3f, normal: vec3f) -> vec3f {
    // use ray incident as seed
    var ray_reflected: vec3f = random_unit_vec(ray_incident);

    // make sure reflection goes outwards.
    if dot(ray_reflected, normal) < 0. {
        ray_reflected = -ray_reflected;
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
    return r.min < n && n < r.max;
}

var<private> spheres: array<Sphere, 2>;
var<private> global_invocation_id: vec2<i32>;

fn world_get_background_color(ray: Ray) -> vec3f {
    // lerp based on y position
    let unit_direction: vec3f = normalize(ray.direction);
    
    let a: f32 = .5 * (unit_direction.y + 1.);
    let pixel_color = (1. - a) * vec3f(1., 1., 1.) + a * vec3f(.5, .7, 1.);
    return pixel_color;
}

fn world_intersect_sphere(ray: Ray, hit_record: ptr<function, HitRecord>, t_range: ptr<function, RangeInclusive>) {
    var i: i32 = 0;

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
        if !is_in_range((*t_range), t) {
            // check with the positive t
            t = (-b + sqrt(det)) / (2. * a);
            if !is_in_range((*t_range), t) {
                continue;
            }
        }

        (*hit_record).p = ray.origin + t * ray.direction;
        (*hit_record).t = t;
        (*hit_record).normal = (-spheres[i].center + (*hit_record).p) / spheres[i].radius;
        (*hit_record).hit = true;
        (*hit_record).material = spheres[i].material;

        (*t_range).max = min((*t_range).max, t);

        continuing {
            i += 1;
        }
    }
}

fn near_zero(v: vec3f) -> bool {
    return length(v) < 0.001;
}

fn num_triangles() -> i32 {
    return i32(arrayLength(&triangles));
}

fn world_intersect_triangle(ray: Ray, hit_record: ptr<function, HitRecord>, t_range: ptr<function, RangeInclusive>) {
    // first treat the triangle as a plane
    var i = 0;
    let num_triangles = num_triangles();
    loop {
        if i >= num_triangles { break; }

        let a = triangles[i].points[0].xyz;
        let b = triangles[i].points[1].xyz;
        let c = triangles[i].points[2].xyz;

        let u = -a + b;
        let v = -a + c;

        let n = cross(u, v);

        let d_dot_n = dot(ray.direction, n);

        // ray is parallel to plane
        if (abs(d_dot_n) < 0.001) {
            continue;
        }

        let t = dot(a - ray.origin, n) / d_dot_n;

        if !is_in_range((*t_range), t) {
            continue;
        }

        // point of intersection
        let p = ray.origin + ray.direction * t;
        
        // create a new coordinate system to check if p is within the bounds of the triangle
        // https://raytracing.github.io/books/RayTracingTheNextWeek.html#quadrilaterals/orientingpointsontheplane 
        // for more details
        let p_small = -a + p;
        let w = n / dot(n, n);

        let alpha = dot(w, cross(u, p_small));
        let beta = dot(w, cross(v, p_small));

        // for a triangle, alpha and beta must both be above or equal to zero and sum to <= 1
        if !(alpha >= 0. && beta >= 0. && alpha + beta <= 1.) {
            continue;
        }

        (*t_range).max = min((*t_range).max, t);

        (*hit_record).t = t;
        (*hit_record).p = p;
        (*hit_record).normal = n;
        (*hit_record).hit = true;
        (*hit_record).material = triangles[i].material;

        continuing {
            i += 1;
        }
    }
}

fn world_get_intersect(ray: Ray, hit_record: ptr<function, HitRecord>) {
    var t_range = RangeInclusive(0.001, f32(0xffffffffu));

    world_intersect_sphere(ray, hit_record, &t_range);
    world_intersect_triangle(ray, hit_record, &t_range);
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

    var hit_record: HitRecord;
    hit_record.hit = false;
    world_get_intersect(ray, &hit_record);

    var out = true;
    if !hit_record.hit {
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

fn world_get_color(ray_0: Ray) -> vec3f {
    // at bottom, completely white
    // at top, sky blue

    // we can't directly take this value for the pixel yet, since our path tracer is going to add that in for us
    var pixel_color = vec3f(0., 0., 0.);
    let max_depth = 5;

    var depth: i32 = 0;
    var pooled_irradiance: vec3f = vec3(1., 1., 1.);
    var ray: Ray = ray_0;

    loop {
        if (depth == max_depth) { break; }

        var hit_record: HitRecord;
        hit_record.hit = false;
        world_get_intersect(ray, &hit_record);
        
        // get color of sky based on ray vector
        if !hit_record.hit {
            pixel_color = pooled_irradiance * world_get_background_color(ray);
            break;
        }
        // make sure this is a color between 0 and 1 lol
        // pixel_color = .5 * (normal + vec3(1., 1., 1.));
        // }

        // // get direct lighting
        // var direct: vec3f = world_get_direct_light_at_point(p);

        // pixel_color = BRDF(ray.direction, new_direction) * ray.normal + direct;

        // update our irradiance
        pooled_irradiance *= hit_record.material.albedo;

        ray.origin = hit_record.p;
        ray.direction = random_in_hemisphere(ray.direction, hit_record.normal) + hit_record.normal;
        if near_zero(ray.direction) {
            ray.direction = hit_record.normal;
        }

        // attenuation *= .5;
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
    spheres[0] = Sphere(vec3(0., -100.5, -1.), 100., Material(vec3f(.5, .5, .5), 0., vec3f(0., 0., 0.), 0.));
    spheres[1] = Sphere(vec3(0., 0., -1.), .5, Material(vec3f(.5, .5, .5), 0., vec3f(0., 0., 0.), 0.));
    // sun
    sun.center = vec3(0., 100., 0.);

    // screen space is [-1., 1.]
    let height = 2. * tan(radians(cam.fov_y) / 2.) * cam.focal_length;
    let width = height * cam.aspect_ratio;

    let camera_coordinate_system: mat3x3f = transpose(mat3x3f(
        cam.right,
        cam.normal,
        -cam.direction,
    ));

    let viewport_u = vec3f(width, 0., 0.) * camera_coordinate_system;
    let viewport_v = vec3f(0., -height, 0.) * camera_coordinate_system;

    // find the top left node
    // treat this screen as if it only exists in two dimensions
    let du = viewport_u / f32(window_size.x);
    let dv = viewport_v / f32(window_size.y);
    let tl_pixel = cam.eye + cam.focal_length * cam.direction - .5 * (viewport_u + viewport_v) + .5 * (du + dv);
    // let tl_pixel = cam.eye - (cam.focal_length * w) - .5 * (viewport_u + viewport_v) + .5 * (du + dv);

    let pixel = tl_pixel + du * f32(GlobalInvocationID.x) + dv * f32(GlobalInvocationID.y);

    let samples = 5;
    // check for intersection of ray with thing
    var i = 0;
    var pixel_color: vec3f = vec3(0., 0., 0.);
    // ray!
    // make sure to center it at the current eye position
    var ray = Ray();
    // this works, but only when we don't rotate the camera at all
    ray.origin = cam.eye;

    // jitter the pixel by +- du / 2, dv / 2
    let du_len = length(du);
    let half_du_len = .5 * du_len;
    let dv_len = length(dv);
    let half_dv_len = .5 * dv_len;

    var seed_out: vec2f = rand22(pixel.xy);
    loop {
        if i == samples { break; }
        // by default should be skybox color
        seed_out = rand22(seed_out) * vec2f(du_len, dv_len) - vec2f(half_du_len, half_dv_len);
        ray.direction = -cam.eye + pixel + vec3f(seed_out.x, seed_out.y, 0.);

        pixel_color += world_get_color(ray);
        continuing {
            i += 1;
        }
    }
    pixel_color /= f32(samples);

    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.));
}
