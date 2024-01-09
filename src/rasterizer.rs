use glam::Vec4;

use crate::*;

// assume intervals inside the bounding box are valid, i.e. x < X
// use vec since 32 is a lot of constants to put on the limited size stack
pub fn to_triangle_coords(bbox: BoundingBox) -> Vec<Vec3> {
    // we will need to describe 8 points of the box
    //
    // good thing is that the bounding box is always axis aligned, so we can
    // just use its length as axis vectors
    let origin = Vec3::new(bbox.x_range[0], bbox.y_range[0], bbox.z_range[0]);
    let x = (bbox.x_range[1] - bbox.x_range[0]) * Vec3::new(1., 0., 0.);
    let y = (bbox.y_range[1] - bbox.y_range[0]) * Vec3::new(0., 1., 0.);
    let z = (bbox.z_range[1] - bbox.z_range[0]) * Vec3::new(0., 0., 1.);

    let points = [
        origin,
        origin + x,
        origin + x + z,
        origin + z,
        origin + y,
        origin + x + y,
        origin + x + y + z,
        origin + y + z,
    ];

    [
        [0_usize, 1, 2, 3],
        [4, 5, 6, 7],
        [2, 6, 7, 3],
        [0, 1, 5, 4],
        [1, 5, 6, 2],
        [4, 0, 3, 7],
    ]
    .iter()
    .flatten()
    .map(|&i| points[i])
    .collect()
}

// assume points can be out of order
// we can't gurantee that the points are in ccw, but that shouldn't matter
pub fn from_quad_to_triangle(q: [Vec3; 4]) -> [Vec3; 6] {
    [q[0], q[1], q[2], q[2], q[3], q[0]]
}
