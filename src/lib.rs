// a context-free OL-system consists of a set of nonterminal symbols, a start symbol, and a set of rules corresponding to each nonterminal symbol.
// there's also a set of defined terminal symbols

use std::collections::HashMap;

use glam::Vec4;
use glam::{Quat, Vec3};

#[derive(Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Material,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct Material {
    pub albedo: Vec3,
    pub padding_1: f32,
    pub ambient: Vec3,
    pub padding_2: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct Triangle {
    pub points: [Vec4; 3],
    pub material: Material,
}
pub struct OLSystem {
    pub rules: HashMap<char, String>,
    pub start: String,
}

// not sure if color_index or diameter will ever be used by me lol
#[derive(Debug, Clone, Copy)]
pub struct Turtle {
    position: Vec3,
    heading: Vec3,
    left: Vec3,
    up: Vec3,
    diameter: f32,
    color_index: u32,
}

// inspired by Algorithm Beauty of Plants, Chapter 1, p. 26 bush algorithm
impl OLSystem {
    pub fn new_bush_system() -> Self {
        let start = String::from("A");
        let rules: HashMap<char, String> = HashMap::from(
            [
                ('A', "[&FL!A]/////'[&FL!A]///////'[&FL!A]"),
                ('F', "S/////F"),
                ('S', "FL"),
                ('L', "['''∧∧{-f+f+f-|-f+f+f}]"),
            ]
            .map(|(s1, s2)| (s1, s2.into())),
        );
        Self { rules, start }
    }

    pub fn generate(&self, generations: u32) -> String {
        let mut b1 = self.start.clone();
        let mut b2 = String::new();

        for _ in 0..generations {
            for char in b1.chars() {
                if let Some(v) = self.rules.get(&char) {
                    b2.push_str(v);
                } else {
                    b2.push(char);
                }
            }

            std::mem::swap(&mut b1, &mut b2);
            b2.clear();
        }

        b1
    }

    pub fn turtle(s: String) -> Vec<Triangle> {
        let mut triangles: Vec<Triangle> = vec![];
        let mut turtle_stack: Vec<Turtle> = vec![];

        let mut current_points: Vec<Vec4> = vec![];
        // let mut current_triangle: Triangle = Triangle {
        //     points: vec![],
        //     material: todo!(),
        // };

        let mut turtle = Turtle {
            position: Vec3::new(0., 0., 0.),
            heading: Vec3::new(0., 1., 0.),
            left: Vec3::new(-1., 0., 0.),
            up: Vec3::new(0., 0., 1.),
            diameter: 1.,
            color_index: 0,
        };
        let delta = 22.5_f32.to_radians();

        s.chars().for_each(|c| match c {
            '[' => {
                turtle_stack.push(turtle);
            }
            ']' => {
                turtle = turtle_stack.pop().expect("turtle stack was empty");
            }
            '{' => {
                // don't care
            }
            'F' => {
                // hopefully 0.1 isn't too small
                turtle.position += 0.1 * turtle.heading;
            }
            'f' => {
                turtle.position += 0.1 * turtle.heading;
                // push position
                current_points.push(Vec4::new(
                    turtle.position.x,
                    turtle.position.y - 1.,
                    turtle.position.z,
                    0.,
                ));
            }
            '}' => {
                // push triangles
                // it's currently in the shape of a rupee, so we only need to
                // use the last coord + 2 outer points to make 4 triangles

                let material = Material {
                    albedo: Vec3::new(0.2, 0.5, 0.1),
                    padding_1: 0.,
                    ambient: Vec3::new(0., 0., 0.),
                    padding_2: 0.,
                };

                for i in 1..=4 {
                    triangles.push(Triangle {
                        points: [
                            // 0
                            current_points[0],
                            // previous
                            current_points[i],
                            // this
                            current_points[i+1]
                        ],
                        material
                    });
                }

                // dbg!(&current_points);

                // (Or, in the future, planes)
                current_points.clear();
            }
            // yaw left
            '+' => {
                rotate_3_around_axis(
                    turtle.up,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    delta,
                );
            }
            // yaw right
            '-' => {
                rotate_3_around_axis(
                    turtle.up,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    -delta,
                );
            }
            // pitch down
            '&' => {
                rotate_3_around_axis(
                    turtle.left,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    delta,
                );
            }
            // pitch up
            '^' => {
                rotate_3_around_axis(
                    turtle.left,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    -delta,
                );
            }
            // roll left
            '\\' => {
                rotate_3_around_axis(
                    turtle.heading,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    delta,
                );
            }
            // roll right
            '/' => {
                rotate_3_around_axis(
                    turtle.heading,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    -delta,
                );
            }
            // turn around
            '|' => {
                rotate_3_around_axis(
                    turtle.up,
                    &mut turtle.heading,
                    &mut turtle.left,
                    &mut turtle.up,
                    180_f32.to_radians(),
                );
            }
            _ => {}
        });
        triangles
    }
}

fn rotate_3_around_axis(axis: Vec3, v1: &mut Vec3, v2: &mut Vec3, v3: &mut Vec3, delta: f32) {
    let n_v1 = rotate_around_axis(*v1, axis, delta);
    let n_v2 = rotate_around_axis(*v2, axis, delta);
    let n_v3 = rotate_around_axis(*v3, axis, delta);

    *v1 = n_v1;
    *v2 = n_v2;
    *v3 = n_v3;
}

fn rotate_around_axis(v: Vec3, axis: Vec3, delta: f32) -> Vec3 {
    let q = Quat::from_axis_angle(axis, delta);
    q.mul_vec3(v)
}
