/// Functions in this file test their .wgsl equivalents to make sure there
/// aren't any bugs
use glam::Vec2;

fn overlap_2(a: Vec2, b: Vec2) -> bool {
    a[0] <= b[1] && b[0] <= a[1]
}

fn overlap_2_w_check(a: Vec2, b: Vec2) -> bool {
    a[0].max(b[0]) <= a[1].min(b[1])
}

fn overlap_3(a: Vec2, b: Vec2, c: Vec2) -> bool {
    a[0].max(b[0]) <= c[1] && a[0].max(c[0]) <= b[1] && b[0].max(c[0]) <= a[1]
}

fn overlap_3_w_check(a: Vec2, b: Vec2, c: Vec2) -> bool {
    a[0].max(b[0]).max(c[0]) <= a[1].min(b[1]).min(c[1])
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::*;

    #[test]
    fn test_overlap_3() {
        assert!(overlap_3(
            Vec2::new(1., 3.),
            Vec2::new(2., 4.),
            Vec2::new(3., 5.)
        ));
        assert!(!overlap_3(
            Vec2::new(1., 2.),
            Vec2::new(3., 4.),
            Vec2::new(5., 6.)
        ));
        assert!(overlap_3_w_check(
            Vec2::new(1., 3.),
            Vec2::new(2., 4.),
            Vec2::new(3., 5.)
        ));
        assert!(!overlap_3_w_check(
            Vec2::new(1., 2.),
            Vec2::new(3., 4.),
            Vec2::new(5., 6.)
        ));
    }

    #[test]
    fn test_overlap_2() {
        assert!(overlap_2(Vec2::new(1., 2.), Vec2::new(2., 3.)));
        assert!(!overlap_2(Vec2::new(1., 2.), Vec2::new(3., 4.)));
        assert!(overlap_2_w_check(Vec2::new(1., 2.), Vec2::new(2., 3.)));
        assert!(!overlap_2_w_check(Vec2::new(1., 2.), Vec2::new(3., 4.)));
    }
}
