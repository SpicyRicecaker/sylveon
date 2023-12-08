@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;

struct Camera {
    eye: vec3f,
    focal_length: f32,
    direction: vec3f,
    aspect_ratio: f32,
    normal: vec3f,
    dummy1: f32,
    right: vec3f,
    dummy2: f32
}

@group(0) @binding(1) var<uniform> cam: Camera;
@group(0) @binding(2) var<uniform> window_size: vec2u;

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

// number of pixels handled by this function
@compute @workgroup_size(1,1,1)
// globalinvocationid: coordinate of current pixel
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let screen_pos: vec2<i32> = vec2(i32(GlobalInvocationID.x), i32(GlobalInvocationID.y));

    var spheres = array<Sphere, 2>(
        Sphere(vec3(0., 0., -3.), .5),
        Sphere(vec3(0., -100.5, -3.), 100.),
    );
    // check for intersection of ray with thing

    // should be uniforms tbh
    let width = 1.;
    let height = width / cam.aspect_ratio;

    // find the top left node
    // treat this screen as if it only exists in two dimensions
    let tl_pixel_corner: vec2<f32> = vec2(-width / 2., height / 2.);
    let du: vec2<f32> = vec2(width / f32(window_size.x), 0.);
    let dv: vec2<f32> = vec2(0., -height / f32(window_size.y));
    let tl_pixel = tl_pixel_corner + du / 2. + dv / 2.;

    let pixel = tl_pixel + du * f32(GlobalInvocationID.x) + dv * f32(GlobalInvocationID.y);

    // ray!
    // make sure to center it at the current eye position
    var bounces_left = 0;

    var origin = cam.eye;
    // this works, but only when we don't rotate the camera at all
    var direction = vec3<f32>(pixel.xy, 0.) + cam.focal_length * cam.direction;
    var pixel_color = vec3f(0., 0., 0.);
    // var multiplier = .5;
    var multiplier = .5;
    var t_min = 100000.;

    loop {
        var hit_record: HitRecord;
        hit_record.hit = false;
        // loop through all objects in scene
        var i: i32 = 0;
        loop {
            if i >= 2 { break; }

            let a = dot(direction, direction);
            let c_o = -spheres[i].center + origin;
            let b = 2. * dot(direction, c_o);
            let c = dot(c_o, c_o) - pow(spheres[i].radius, 2.);

            let det = pow(b, 2.) - 4. * a * c;

            if det < 0. {
                continue;
            }

            // why not always take smallest t?
            let t = (-b - sqrt(det)) / 2. * a;

            if t > t_min {
                continue;
            }

            let p = origin + t * direction;
            let normal = normalize(-spheres[i].center + p);

            hit_record.p = p;
            hit_record.t = t;
            t_min = t;
            hit_record.normal = normal;
            hit_record.hit = true;

            // make sure this is a color between 0 and 1 lol
            // pixel_color = multiplier * (normal + vec3(1., 1., 1.)) / 2.;
            pixel_color = multiplier * (normal + vec3(1., 1., 1.)) / 2.;

            multiplier *= .5;

            origin = hit_record.p;
            direction = random_in_hemisphere(direction, hit_record.normal);

            continuing {
                i = i + 1;
            }
        }

        if !hit_record.hit {
            break;
        } 
        
        bounces_left = bounces_left - 1;

        if bounces_left <= 0 { break; }
    }


    textureStore(color_buffer, screen_pos, vec4<f32>(pixel_color, 1.));
}
