// a context-free OL-system consists of a set of nonterminal symbols, a start symbol, and a set of rules corresponding to each nonterminal symbol.
// there's also a set of defined terminal symbols

use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
};

use glam::{Quat, Vec3, Mat4};
use glam::{Vec2, Vec4};
use rand::{thread_rng, Rng};

#[cfg(test)]
mod wgsl_test;
pub mod rasterizer;

#[derive(Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Material,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
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
                            current_points[i + 1],
                        ],
                        material,
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

fn aabb_triangle(triangle: &Triangle) -> BoundingBox {
    AccelStruct::get_bounding_box(&[*triangle])
}

// true if in the proper order, i.e. a is less than or equal to b
pub fn compare_aabb_by_axis(a: BoundingBox, b: BoundingBox, axis: usize) -> Ordering {
    if let Some(ordering) =
        center_of_bounding_box(a)[axis].partial_cmp(&center_of_bounding_box(b)[axis])
    {
        ordering
    } else {
        Ordering::Equal
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct BoundingBox {
    pub x_range: Vec2,
    pub y_range: Vec2,
    pub z_range: Vec2,
    pub _1: Vec2,
}

// assume that there isn't a 0 0 range
fn center_of_bounding_box(bb: BoundingBox) -> Vec3 {
    Vec3::new(
        (bb.x_range[1] + bb.x_range[0]) / 2.,
        (bb.y_range[1] + bb.y_range[0]) / 2.,
        (bb.z_range[1] + bb.z_range[0]) / 2.,
    )
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct AccelStruct {
    pub tree: Vec<Primitive>,
}

pub struct WorklistElement {
    pub depth: usize,
    pub range: (usize, usize),
}

impl AccelStruct {
    // triangles is guaranteed to be nonzero
    fn get_bounding_box(triangles: &[Triangle]) -> BoundingBox {
        let mut x_range = Vec2::new(triangles[0].points[0].x, triangles[0].points[0].x);
        let mut y_range = Vec2::new(triangles[0].points[0].y, triangles[0].points[0].y);
        let mut z_range = Vec2::new(triangles[0].points[0].z, triangles[0].points[0].z);

        triangles
            .iter()
            .flat_map(|&t| t.points.into_iter())
            .for_each(|t| {
                x_range[0] = x_range[0].min(t.x);
                x_range[1] = x_range[1].max(t.x);
                y_range[0] = y_range[0].min(t.y);
                y_range[1] = y_range[1].max(t.y);
                z_range[0] = z_range[0].min(t.z);
                z_range[1] = z_range[1].max(t.z);
            });
        BoundingBox {
            x_range,
            y_range,
            z_range,
            ..Default::default()
        }
    }

    fn _dbg_bounding_box() -> BoundingBox {
        BoundingBox {
            x_range: Vec2::new(0., 0.1),
            y_range: Vec2::new(0., 0.1),
            z_range: Vec2::new(0., 0.1),
            ..Default::default()
        }
    }

    pub fn new(triangles: &[Triangle]) -> (Self, Vec<BoundingBox>) {
        let mut worklist: VecDeque<WorklistElement> = VecDeque::new();
        // assume triangle size is greater than 1
        worklist.push_back(WorklistElement {
            depth: 0,
            range: (0, triangles.len() - 1),
        });

        let mut triangle_indices = vec![];
        for i in 0..triangles.len() {
            triangle_indices.push(i);
        }

        let mut bounding_boxes = vec![];
        let mut tree = vec![];

        // calculate the maximum depth required
        let max_depth = (triangles.len() as f64).log2().ceil() as usize;

        // _dbg
        // {
        //     bounding_boxes.push(AccelStruct::_dbg_bounding_box());
        //     nodes.push(Primitive::from_bounding_box_ptr(
        //         (bounding_boxes.len() - 1) as i32,
        //     ));
        // }

        // build a binary tree, where all nodes not at the last layer make up a perfect binary tree,
        // and nodes at the last layer are placed arbitrarily

        let mut rng = thread_rng();

        // range is inclusive
        // i know for sure this can be made more concise but this is what I came up with for now
        while let Some(WorklistElement {
            depth,
            range: (p, q),
        }) = worklist.pop_front()
        {
            let n = q - p + 1;

            match max_depth - depth {
                0 => {
                    // add two triangles instead
                    assert_eq!(p, q);
                    tree.push(Primitive::from_triangle_ptr(triangle_indices[p] as i32));
                }
                diff => {
                    // insert the bounding boxes into the scene
                    bounding_boxes.push(AccelStruct::get_bounding_box(
                        &triangle_indices[p..=q]
                            .iter()
                            .map(|&i| triangles[i])
                            .collect::<Vec<_>>(),
                    ));

                    tree.push(Primitive::from_bounding_box_ptr(
                        (bounding_boxes.len() - 1) as i32,
                    ));

                    // if we're at the last boundary box before leaves, and this
                    // boundary box only has one child, then just clone the child to
                    // make sure the binary tree is filled
                    if diff == 1 && n == 1 {
                        // problem: r + 1 is greater than q when p = 1, q = 1
                        worklist.push_back(WorklistElement {
                            depth: depth + 1,
                            range: (p, q),
                        });
                        worklist.push_back(WorklistElement {
                            depth: depth + 1,
                            range: (p, q),
                        });
                    } else {
                        // sort triangle indices by a random axis.
                        // actually, don't sort for now, since for some reason
                        // not sorting gives better performance

                        // triangle_indices[p..=q].sort_by(|a, b| {
                        //     compare_aabb_by_axis(
                        //         aabb_triangle(&triangles[*a]),
                        //         aabb_triangle(&triangles[*b]),
                        //         rng.gen_range(0..3),
                        //     )
                        // });

                        let r = (q - p) / 2 + p;
                        // the way that we built the tree, the last layer is made completely
                        // of leaves
                        worklist.push_back(WorklistElement {
                            depth: depth + 1,
                            range: (p, r),
                        });
                        worklist.push_back(WorklistElement {
                            depth: depth + 1,
                            range: (r + 1, q),
                        });
                    }
                }
            }
        }

        dbg!(tree.len());
        dbg!(triangles.len());

        dbg!((0..tree.len())
            .filter(|&i| { tree[i].primitive_type == PrimitiveType::BoundingBox })
            .collect::<Vec<_>>()
            .len());

        (Self { tree }, bounding_boxes)
    }
}

// u32
#[repr(i32)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, bytemuck::NoUninit)]
pub enum PrimitiveType {
    #[default]
    Triangle = 0,
    Sphere = 1,
    BoundingBox = 2,
}

// A vec4 is 32 bytes wide
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct Primitive {
    // all
    pub primitive_type: PrimitiveType,
    pub pointer: i32,
}

impl Primitive {
    fn from_triangle_ptr(pointer: i32) -> Self {
        Self {
            primitive_type: PrimitiveType::Triangle,
            pointer,
            ..Default::default()
        }
    }
    fn from_bounding_box_ptr(pointer: i32) -> Self {
        Self {
            primitive_type: PrimitiveType::BoundingBox,
            pointer,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_build_accel_struct() {
        let bush_system = OLSystem::new_bush_system();
        let generations = 5;
        let s = bush_system.generate(generations);
        let triangles = OLSystem::turtle(s);

        let (accel_struct, _bounding_boxes) = AccelStruct::new(&triangles);

        let mut set = HashSet::new();

        accel_struct
            .tree
            .into_iter()
            .filter(|p| p.primitive_type == PrimitiveType::Triangle)
            .for_each(|p| {
                set.insert(p.pointer);
            });

        dbg!(set.len(), triangles.len());
        assert!(set.len() == triangles.len());
    }
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default)]

pub struct Camera {
    pub eye: Vec3,
    pub focal_length: f32,
    pub direction: Vec3,
    pub aspect_ratio: f32,
    pub up: Vec3,
    pub fov_y: f32,
    pub right: Vec3,
    pub _1: f32,
}

impl Camera {
    pub fn projection_matrix(&self) -> Mat4 {
        // projection
        Mat4::perspective_rh_gl(self.fov_y.to_radians(), self.aspect_ratio, 0., 10000.)
    }

    pub fn view_matrix(&self) -> Mat4 {
        // view matrix
        Mat4::look_to_rh(self.eye, self.direction, self.up)
    }
}