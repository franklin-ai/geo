use std::{fmt, fmt::Debug};

use crate::coord;
use crate::{BoundingRect, Coord, GeoFloat, Line, MultiPoint, Polygon, Triangle};
use ndarray_linalg::solve::Determinant;

pub const DEFAULT_SUPER_TRIANGLE_EXPANSION: f64 = 20.;

type Result<T> = std::result::Result<T, DelaunayTriangulationError>;

/// Delaunay Triangulation for a given set of points using the
/// [Bowyer](https://doi.org/10.1093%2Fcomjnl%2F24.2.162)-[Watson](https://doi.org/10.1093%2Fcomjnl%2F24.2.167)
/// algorithm
pub trait TriangulateDelaunay<T: GeoFloat> {
    /// # Examples
    ///
    /// ```
    /// use geo::{coord, polygon, Triangle, TriangulateDelaunay};
    ///
    /// let points = polygon![
    ///     (x: 0., y: 0.),
    ///     (x: 1., y: 2.),
    ///     (x: 2., y: 4.),
    ///     (x: 2., y: 0.),
    ///     (x: 3., y: 2.),
    ///     (x: 4., y: 0.),
    /// ];
    ///
    /// let tri_force = points.delaunay_triangulation().unwrap();
    ///
    /// assert_eq!(vec![
    ///     Triangle(
    ///         coord! {x: 1., y: 2.},
    ///         coord! {x: 0., y: 0.},
    ///         coord! {x: 2., y: 0.},
    ///     ),
    ///     Triangle(
    ///         coord! {x: 2., y: 4.},
    ///         coord! {x: 1., y: 2.},
    ///         coord! {x: 3., y: 2.},
    ///     ),
    ///     Triangle(
    ///         coord! {x: 1., y: 2.},
    ///         coord! {x: 2., y: 0.},
    ///         coord! {x: 3., y: 2.},
    ///     ),
    ///     Triangle(
    ///         coord! {x: 3., y: 2.},
    ///         coord! {x: 2., y: 0.},
    ///         coord! {x: 4., y: 0.},
    ///     )],
    ///     tri_force
    /// );
    ///
    /// ```
    ///
    fn delaunay_triangulation(&self) -> Result<Vec<Triangle<T>>>;
}

impl<T: GeoFloat> TriangulateDelaunay<T> for Polygon<T>
where
    f64: From<T>,
{
    fn delaunay_triangulation(&self) -> Result<Vec<Triangle<T>>> {
        let super_triangle = DelaunayTriangle(create_super_triangle(self)?);

        let mut triangles: Vec<DelaunayTriangle<T>> = vec![super_triangle.clone()];
        for pt in self.exterior().coords() {
            add_point(&mut triangles, pt)?;
        }
        remove_super_triangle(&mut triangles, &super_triangle);
        Ok(triangles.iter().map(|x| x.0).collect())
    }
}

impl<T: GeoFloat> TriangulateDelaunay<T> for MultiPoint<T>
where
    f64: From<T>,
{
    fn delaunay_triangulation(&self) -> Result<Vec<Triangle<T>>> {
        let poly = Polygon::new(self.into_iter().cloned().collect(), vec![]);
        poly.delaunay_triangulation()
    }
}

fn create_super_triangle<T: GeoFloat>(geometry: &Polygon<T>) -> Result<Triangle<T>>
where
    f64: From<T>,
{
    let expand_factor = T::from(DEFAULT_SUPER_TRIANGLE_EXPANSION)
        .ok_or(DelaunayTriangulationError::FailedToConstructSuperTriangle)?;
    let bounds = geometry
        .bounding_rect()
        .ok_or(DelaunayTriangulationError::FailedToConstructSuperTriangle)?;
    let width = bounds.width() * expand_factor;
    let height = bounds.height() * expand_factor;
    let bounds_min = bounds.min();
    let bounds_max = bounds.max();

    Ok(Triangle(
        coord! {x: bounds_min.x - width, y: bounds_min.y},
        coord! {x: bounds_max.x + width, y: bounds_min.y - height},
        coord! {x: bounds_max.x + width, y: bounds_max.y + height},
    ))
}

fn add_point<T: GeoFloat>(triangles: &mut Vec<DelaunayTriangle<T>>, c: &Coord<T>) -> Result<()>
where
    f64: From<T>,
{
    let mut bad_triangles: Vec<&DelaunayTriangle<T>> = Vec::new();

    // Check for the triangles where the point is present within the
    // corresponding circumcircle.
    for tri in triangles.iter() {
        if tri.is_in_circumcircle(c)? {
            bad_triangles.push(tri);
        }
    }

    let mut polygon: Vec<Line<T>> = Vec::new();

    // Find all edges that are not shared with
    // other triangles, these can be removed.
    for tri in bad_triangles.iter() {
        for edge in tri.0.to_lines().iter() {
            let mut shared_edge = false;
            for other_tri in bad_triangles.iter() {
                if tri == other_tri {
                    continue;
                }

                for other_edge in other_tri.0.to_lines().iter() {
                    if is_line_shared(edge, other_edge) {
                        shared_edge = true;
                    }
                }
            }

            if !shared_edge {
                polygon.push(*edge);
            }
        }
    }

    // Remove the bad triangles
    let mut new_triangles: Vec<DelaunayTriangle<T>> = triangles
        .iter()
        .filter(|x| !bad_triangles.contains(x))
        .cloned()
        .collect();

    polygon
        .iter()
        .for_each(|x| new_triangles.push(DelaunayTriangle(Triangle(x.start, x.end, *c))));

    // Replace the triangles
    triangles.clear();
    new_triangles.iter().for_each(|x| triangles.push(x.clone()));

    Ok(())
}

fn remove_super_triangle<T: GeoFloat>(
    triangles: &mut Vec<DelaunayTriangle<T>>,
    super_triangle: &DelaunayTriangle<T>,
) {
    let mut new_triangles: Vec<DelaunayTriangle<T>> = Vec::new();
    let super_tri_vertices = super_triangle.0.to_array();

    for tri in triangles.iter() {
        let mut add_to_update = true;
        for pt in tri.0.to_array().iter() {
            if super_tri_vertices.contains(pt) {
                add_to_update = false;
            }
        }
        if add_to_update {
            new_triangles.push(tri.clone())
        }
    }

    triangles.clear();
    new_triangles.iter().for_each(|x| triangles.push(x.clone()));
}

/// A Circle defined by a centre and a radius
///
/// # Examples
///
/// ```rust
/// use geo_types::coord;
/// use geo::triangulate_delaunay::Circle;
///
/// let circle = Circle::new(
///     coord! {x: 10., y: 2.},
///     12.
/// );
/// assert_eq!(circle.center, coord! {x:10., y: 2.});
/// assert_eq!(circle.radius, 12.);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Circle<T: GeoFloat> {
    pub center: Coord<T>,
    pub radius: T,
}

impl<T: GeoFloat> Circle<T> {
    pub fn new(center: Coord<T>, radius: T) -> Self {
        Self { center, radius }
    }
}

/// A triangle structure used during Delaunay Triangulation
#[derive(Debug, Clone, PartialEq)]
pub struct DelaunayTriangle<T: GeoFloat>(Triangle<T>);

impl<T: GeoFloat> From<Triangle<T>> for DelaunayTriangle<T> {
    fn from(value: Triangle<T>) -> Self {
        DelaunayTriangle(value)
    }
}

impl<T: GeoFloat> From<DelaunayTriangle<T>> for Triangle<T> {
    fn from(value: DelaunayTriangle<T>) -> Self {
        value.0
    }
}

fn is_line_shared<T: GeoFloat>(a: &Line<T>, b: &Line<T>) -> bool {
    (a.start == b.start && a.end == b.end) || (a.start == b.end && a.end == b.start)
}

/// Methods required for delaunay triangulation
impl<T: GeoFloat> DelaunayTriangle<T>
where
    f64: From<T>,
{
    #[cfg(feature = "voronoi")]
    /// Check if a `DelaunayTriangle` shares at least one vertex
    pub fn shares_vertex(&self, other: &DelaunayTriangle<T>) -> bool {
        let other_vertices = other.0.to_array();
        for vertex in self.0.to_array().iter() {
            if other_vertices.contains(vertex) {
                return true;
            }
        }
        false
    }

    #[cfg(feature = "voronoi")]
    /// Check if a `DelaunayTriangle` is a neighbour i.e.
    /// shares an edge.
    /// If the triangles are neighbours the shared edge is returned
    pub fn shares_edge(&self, other: &DelaunayTriangle<T>) -> Option<Line<T>> {
        let other_lines = other.0.to_lines();
        for line in self.0.to_lines().iter() {
            for other_line in other_lines.iter() {
                if is_line_shared(line, other_line) {
                    return Some(*line);
                }
            }
        }
        None
    }

    /// Check if a `Coord` is in the [Circumcircle](https://en.wikipedia.org/wiki/Circumcircle)
    /// for the Delaunay triangle.
    /// This method uses the determinant of the vertices of the triangle and the
    /// new point as described by [Guibas & Stolfi](https://doi.org/10.1145%2F282918.282923)
    /// and on [Wikipedia](https://en.wikipedia.org/wiki/Delaunay_triangulation#Algorithms).
    pub fn is_in_circumcircle(&self, c: &Coord<T>) -> Result<bool> {
        let a_d_x: f64 = (self.0 .0.x - c.x).into();
        let a_d_y: f64 = (self.0 .0.y - c.y).into();
        let b_d_x: f64 = (self.0 .1.x - c.x).into();
        let b_d_y: f64 = (self.0 .1.y - c.y).into();
        let c_d_x: f64 = (self.0 .2.x - c.x).into();
        let c_d_y: f64 = (self.0 .2.y - c.y).into();

        let eqn_sys = ndarray::arr2(&[
            [a_d_x, a_d_y, a_d_x.powi(2) + a_d_y.powi(2)],
            [b_d_x, b_d_y, b_d_x.powi(2) + b_d_y.powi(2)],
            [c_d_x, c_d_y, c_d_x.powi(2) + c_d_y.powi(2)],
        ]);

        Ok(eqn_sys
            .det()
            .map_err(|_| DelaunayTriangulationError::FailedToCheckPointInCircumcircle)?
            > 0.0)
    }

    #[cfg(feature = "voronoi")]
    /// Get the [Circumcircle](https://en.wikipedia.org/wiki/Circumcircle)
    /// for the Delaunay triangle.
    pub fn get_circumcircle(&self) -> Result<Circle<T>> {
        // Pin the triangle to the origin to simplify the calculation
        let b = self.0 .1 - self.0 .0;
        let c = self.0 .2 - self.0 .0;

        let a_x = self.0 .0.x;
        let a_y = self.0 .0.y;
        let b_x = b.x;
        let b_y = b.y;
        let c_x = c.x;
        let c_y = c.y;

        let d = T::from(2.0).ok_or(DelaunayTriangulationError::GeoTypeConversionError)?
            * (b_x * c_y - b_y * c_x);

        if d.is_zero() {
            return Err(DelaunayTriangulationError::FailedToComputeCircumcircle);
        }

        let u_x = (c_y * (b_x.powi(2) + b_y.powi(2)) - b_y * (c_x.powi(2) + c_y.powi(2))) / d;
        let u_y = (b_x * (c_x.powi(2) + c_y.powi(2)) - c_x * (b_x.powi(2) + b_y.powi(2))) / d;

        let radius = T::sqrt(u_x.powi(2) + u_y.powi(2));

        Ok(Circle {
            center: coord! {x: a_x + u_x, y: a_y + u_y},
            radius,
        })
    }
}

/// Delaunay Triangulation Errors
#[derive(Debug, PartialEq, Eq)]
pub enum DelaunayTriangulationError {
    /// Failed to compute the circumcircle for a given triangle.
    /// This can occur if the points are collinear.
    FailedToComputeCircumcircle,
    /// Failed to check if a point is in a circumcircle.
    FailedToCheckPointInCircumcircle,
    /// Failed to construct the super triangle.
    /// This error occurs when the `Polygon` describing the points to
    /// triangulate does not return a bounding rectangle.
    FailedToConstructSuperTriangle,
    FailedToConvertSuperTriangleFactor,
    GeoTypeConversionError,
}

impl fmt::Display for DelaunayTriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DelaunayTriangulationError::FailedToComputeCircumcircle => {
                write!(f, "Cannot compute circumcircle.")
            }
            DelaunayTriangulationError::FailedToCheckPointInCircumcircle => {
                write!(f, "Cannot determine if the point is in the circumcircle.")
            }
            DelaunayTriangulationError::FailedToConstructSuperTriangle => {
                write!(f, "Failed to construct the super triangle.")
            }
            DelaunayTriangulationError::GeoTypeConversionError => {
                write!(f, "Failed to convert from Geo type T")
            }
            DelaunayTriangulationError::FailedToConvertSuperTriangleFactor => {
                write!(f, "Failed to convert super triangle expansion factor")
            }
        }
    }
}

#[cfg(test)]
mod test {

    use geo_types::{LineString, MultiPoint, Point};

    use crate::Contains;

    use super::*;

    #[test]
    fn test_triangle_shares_vertex() {
        let triangle = DelaunayTriangle(Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 10., y: 20.},
            coord! {x: -12., y: -2.},
        ));
        let other = DelaunayTriangle(Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 30., y: 40.},
            coord! {x: 40., y: 30.},
        ));

        assert!(triangle.shares_vertex(&other));

        let other = DelaunayTriangle(Triangle::new(
            coord! {x: 30., y: 40.},
            coord! {x: 40., y: 30.},
            coord! {x: 50., y: 20.},
        ));

        assert!(!triangle.shares_vertex(&other));
    }

    #[test]
    fn test_triangle_is_neighbour() {
        let triangle = DelaunayTriangle(Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 10., y: 20.},
            coord! {x: -12., y: -2.},
        ));
        let other = DelaunayTriangle(Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 10., y: 20.},
            coord! {x: 30., y: 40.},
        ));

        assert_eq!(
            triangle.shares_edge(&other).unwrap(),
            Line::new(coord! {x: 0., y:0.}, coord! { x: 10., y: 20.})
        );

        let other = DelaunayTriangle(Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 30., y: 40.},
            coord! {x: 40., y: 50.},
        ));

        assert!(triangle.shares_edge(&other).is_none());
    }

    #[test]
    fn test_point_in_circumcircle() {
        let triangle = DelaunayTriangle(Triangle::new(
            coord! {x: 10., y: 10.},
            coord! {x: 30., y: 10.},
            coord! {x: 20., y: 20.},
        ));

        assert!(triangle
            .is_in_circumcircle(&coord! {x: 20., y: 10.})
            .unwrap());
        assert!(!triangle
            .is_in_circumcircle(&coord! {x: 10., y: 30.})
            .unwrap());
    }

    #[test]
    fn test_get_circumcircle() {
        let triangle = DelaunayTriangle(Triangle::new(
            coord! {x: 10., y: 10.},
            coord! {x: 20., y: 20.},
            coord! {x: 30., y: 10.},
        ));

        let circle = triangle.get_circumcircle().unwrap();
        approx::assert_relative_eq!(circle.center, coord! {x: 20., y: 10.});
        approx::assert_relative_eq!(circle.radius, 10.);
    }

    #[test]
    fn test_get_circumcircle_collinear_points() {
        let triangle = DelaunayTriangle(Triangle::new(
            coord! {x: 10., y: 10.},
            coord! {x: 20., y: 20.},
            coord! {x: 30., y: 30.},
        ));

        // The circumcircle for collinear points cannot be
        // determined as the radius would be infinite
        triangle
            .get_circumcircle()
            .expect_err("Cannot compute circumcircle");
    }

    #[test]
    fn test_get_super_triangle() {
        let points: Polygon = Polygon::new(
            LineString::from(vec![(0., 0.), (1., 0.), (1., 1.), (0., 1.)]),
            vec![],
        );

        let super_tri = create_super_triangle(&points).unwrap();
        assert!(super_tri.contains(&points));
    }

    #[test]
    fn test_add_point() {
        // Create a super triangle
        let mut triangles = vec![DelaunayTriangle(Triangle::new(
            coord! {x: -20., y: 0.},
            coord! {x: 21., y: -20.},
            coord! {x: 21., y: 21.},
        ))];

        let expected_result = vec![
            DelaunayTriangle(Triangle::new(
                coord! {x: -20., y: 0.},
                coord! {x: 21., y: -20.},
                coord! {x: 0., y: 0.},
            )),
            DelaunayTriangle(Triangle::new(
                coord! {x: 21., y: -20.},
                coord! {x: 21., y: 21.},
                coord! {x: 0., y: 0.},
            )),
            DelaunayTriangle(Triangle::new(
                coord! {x: 21., y: 21.},
                coord! {x: -20., y: 0.},
                coord! {x: 0., y: 0.},
            )),
        ];

        add_point(&mut triangles, &coord! {x: 0., y: 0.}).unwrap();

        assert_eq!(expected_result, triangles);
    }

    //Execute the geos tests

    // https://github.com/libgeos/geos/blob/d51982c6da5b7adb63ca0933ae7b53828cc8d72e/tests/unit/triangulate/DelaunayTest.cpp#L113
    #[test]
    fn test_triangle() {
        let points = MultiPoint::new(vec![
            Point::new(10., 10.),
            Point::new(10., 20.),
            Point::new(20., 20.),
        ]);

        let expected_triangle = vec![Triangle::new(
            coord! {x: 10.0, y: 20.},
            coord! {x: 10.0, y: 10.},
            coord! {x: 20.0, y: 20.},
        )];

        assert_eq!(points.delaunay_triangulation().unwrap(), expected_triangle);
    }

    // https://github.com/libgeos/geos/blob/d51982c6da5b7adb63ca0933ae7b53828cc8d72e/tests/unit/triangulate/DelaunayTest.cpp#L127
    #[test]
    fn test_random() {
        let points = MultiPoint::new(vec![
            Point::new(50., 40.),
            Point::new(140., 70.),
            Point::new(80., 100.),
            Point::new(130., 140.),
            Point::new(30., 150.),
            Point::new(70., 180.),
            Point::new(190., 110.),
            Point::new(120., 20.),
        ]);

        let expected_triangles = vec![
            Triangle::new(
                coord! {x: 50.0, y: 40.},
                coord! {x: 80.0, y: 100.},
                coord! {x: 30.0, y: 150.},
            ),
            Triangle::new(
                coord! {x: 30.0, y: 150.},
                coord! {x: 80.0, y: 100.},
                coord! {x: 70.0, y: 180.},
            ),
            Triangle::new(
                coord! {x: 80.0, y: 100.},
                coord! {x: 130.0, y: 140.},
                coord! {x: 70.0, y: 180.},
            ),
            Triangle::new(
                coord! {x: 70.0, y: 180.},
                coord! {x: 130.0, y: 140.},
                coord! {x: 190.0, y: 110.},
            ),
            Triangle::new(
                coord! {x: 130.0, y: 140.},
                coord! {x: 140.0, y: 70.},
                coord! {x: 190.0, y: 110.},
            ),
            Triangle::new(
                coord! {x: 190.0, y: 110.},
                coord! {x: 140.0, y: 70.},
                coord! {x: 120.0, y: 20.},
            ),
            Triangle::new(
                coord! {x: 140.0, y: 70.},
                coord! {x: 80.0, y: 100.},
                coord! {x: 120.0, y: 20.},
            ),
            Triangle::new(
                coord! {x: 80.0, y: 100.},
                coord! {x: 50.0, y: 40.},
                coord! {x: 120.0, y: 20.},
            ),
            Triangle::new(
                coord! {x: 80.0, y: 100.},
                coord! {x: 140.0, y: 70.},
                coord! {x: 130.0, y: 140.},
            ),
        ];

        let delaunay_triangles = points.delaunay_triangulation().unwrap();

        assert_eq!(delaunay_triangles.len(), expected_triangles.len());
        for tri in delaunay_triangles.iter() {
            assert!(expected_triangles.contains(tri));
        }
    }

    // Test grid
    // https://github.com/libgeos/geos/blob/d51982c6da5b7adb63ca0933ae7b53828cc8d72e/tests/unit/triangulate/DelaunayTest.cpp#L143
    #[test]
    fn test_grid() {
        let points = MultiPoint::new(vec![
            Point::new(10., 10.),
            Point::new(10., 20.),
            Point::new(20., 20.),
            Point::new(20., 10.),
            Point::new(10., 0.),
            Point::new(0., 0.),
            Point::new(0., 10.),
            Point::new(0., 20.),
        ]);

        let expected_triangles = vec![
            Triangle::new(
                coord! {x: 0.0, y: 0.},
                coord! {x: 10.0, y: 10.},
                coord! {x: 0.0, y: 10.},
            ),
            Triangle::new(
                coord! {x: 10.0, y: 0.},
                coord! {x: 10.0, y: 10.},
                coord! {x: 0.0, y: 0.},
            ),
            Triangle::new(
                coord! {x: 0.0, y: 10.},
                coord! {x: 10.0, y: 20.},
                coord! {x: 0.0, y: 20.},
            ),
            Triangle::new(
                coord! {x: 10.0, y: 10.},
                coord! {x: 10.0, y: 20.},
                coord! {x: 0.0, y: 10.},
            ),
            Triangle::new(
                coord! {x: 10.0, y: 20.},
                coord! {x: 10.0, y: 10.},
                coord! {x: 20.0, y: 20.},
            ),
            Triangle::new(
                coord! {x: 20.0, y: 20.},
                coord! {x: 10.0, y: 10.},
                coord! {x: 20.0, y: 10.},
            ),
            Triangle::new(
                coord! {x: 20.0, y: 10.},
                coord! {x: 10.0, y: 10.},
                coord! {x: 10.0, y: 0.},
            ),
        ];

        let delaunay_triangles = points.delaunay_triangulation().unwrap();

        assert_eq!(delaunay_triangles.len(), expected_triangles.len());
        for tri in delaunay_triangles.iter() {
            assert!(expected_triangles.contains(tri));
        }
    }
}
