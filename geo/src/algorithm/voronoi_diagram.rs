use core::num;
use std::{fmt, result};

use geo_types::{coord, Coord, CoordFloat, Line, Point, Polygon, Rect, Triangle};

use crate::algorithm::euclidean_distance::EuclideanDistance;
use crate::triangulate_delaunay::{DelaunayTriangle, DelaunayTriangulationError};
use crate::{BoundingRect, GeoFloat};

type Result<T> = result::Result<T, VoronoiDiagramError>;

pub trait VoronoiDiagram<T: CoordFloat> {
    fn compute_voronoi(&self) -> Result<Vec<Polygon<T>>>;
}

/// Compute the Voronoi Diagram from Delaunay Triangles
/// The Voronoi Diagram is a [dual graph](https://en.wikipedia.org/wiki/Dual_graph)
/// of the [Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
/// and thus the Voronoi Diagram can be created from the Delaunay Triangulation.
pub fn compute_voronoi_from_delaunay<T: CoordFloat>(
    triangles: &Vec<Triangle<T>>,
    clipping_mask: &Option<Polygon<T>>,
) -> Result<Vec<Polygon<T>>>
where
    f64: From<T>,
{
    let delaunay_triangles: Vec<DelaunayTriangle<T>> =
        triangles.iter().map(|x| x.clone().into()).collect();

    // The centres of the delaunay circumcircles form the vertices of the
    // voronoi diagram
    let mut vertices: Vec<Coord> = Vec::new();
    for tri in delaunay_triangles.iter() {
        vertices.push(
            tri.get_circumcircle()
                .map_err(|e| VoronoiDiagramError::DelaunayError(e))?
                .center,
        );
    }
    let (neighbours, mut shared_edges) = find_shared_edges(&delaunay_triangles);

    Ok(vec![])
}

fn find_shared_edges<T: CoordFloat>(
    triangles: &Vec<DelaunayTriangle<T>>,
) -> (Vec<Vec<Option<usize>>>, Vec<Line<T>>)
where
    f64: From<T>,
{
    let mut neighbours: Vec<Vec<Option<usize>>> = Vec::new();
    let mut shared_edges: Vec<Line<T>> = Vec::new();

    // Search the delaunay triangles for neighbour triangles and shared edges
    for (tri_idx, tri) in triangles.iter().enumerate() {
        for (other_idx, other_tri) in triangles.iter().enumerate() {
            if tri_idx == other_idx {
                continue;
            }

            if let Some(shared_edge) = tri.shares_edge(&other_tri) {
                if !neighbours.contains(&vec![Some(tri_idx), Some(other_idx)])
                    && !neighbours.contains(&vec![Some(other_idx), Some(tri_idx)])
                {
                    neighbours.push(vec![Some(tri_idx), Some(other_idx)]);
                }

                if !shared_edges.contains(&shared_edge) {
                    shared_edges.push(shared_edge);
                }
            }
        }
    }

    // For Voronoi diagrams, the triangles / circumcenters that are on the edge of the
    // diagram require connections to infinity to ensure separation of points between
    // voronoi cells.
    // These connections to infinity will be bounded later
    // Add the connections from infinity
    for idx in 0..triangles.len() {
        let num_neighbours = neighbours[idx].len();
        for _ in 0..3 - num_neighbours {
            neighbours.push(vec![None, Some(idx)]);
        }
    }

    (neighbours, shared_edges)
}

fn get_perpendicular_line<T: CoordFloat>(line: &Line<T>) -> Result<Line<T>>
where
    f64: From<T>,
{
    let slope = f64::from(line.slope());

    // Vertical line
    if slope.is_infinite() {
        return Ok(Line::new(
            coord! {x: line.start.x, y: line.start.y},
            coord! {x: line.start.x - line.dy(), y: line.start.y},
        ));
    } else if slope == 0. {
        return Ok(Line::new(
            coord! {x: line.start.x, y: line.start.y},
            coord! {x: line.start.x, y: line.start.y + line.dx()},
        ));
    } else {
        let midpoint = coord! {
            x: f64::from(line.start.x) + f64::from(line.dx()) / 2.,
            y: f64::from(line.start.y) + f64::from(line.dy()) / 2.,
        };
        let m = -1. / slope;
        // y = mx + b
        let b = midpoint.y - m * midpoint.x;
        let x = midpoint.x + f64::from(line.dx());
        let y = m * x + b;
        let x = T::from(x).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?;
        let y = T::from(y).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?;
        let start_x =
            T::from(midpoint.x).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?;
        let start_y =
            T::from(midpoint.y).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?;
        return Ok(Line::new(
            coord! {x: start_x, y: start_y},
            coord! {x: x, y: y},
        ));
    }
}

fn move_line_to_circumcenter<T: CoordFloat>(
    line: &Line<T>,
    start_coord: &Coord<T>,
    center_coord: &Coord<T>,
) -> Line<T>
where
    f64: From<T>,
{
    let slope = line.slope();

    // Lines need to move away from the circumcentre and travel
    // towards infinity
    let end = if slope.is_infinite() {
        // It is a vertical line so we need to ensure it moves up or down correctly
        if start_coord.y < center_coord.y {
            coord! {x: start_coord.x, y: start_coord.y - line.dy().abs()}
        } else {
            coord! {x: start_coord.x, y: start_coord.y + line.dy().abs()}
        }
    } else {
        // A sloping or horizontal line
        let b = start_coord.y - line.slope() * start_coord.x;
        let end_x = if start_coord.x < center_coord.x {
            start_coord.x - line.dx().abs()
        } else {
            start_coord.x + line.dx().abs()
        };
        let end_y = line.slope() * end_x + b;

        coord! {x: end_x, y: end_y}
    };

    return Line::new(start_coord.clone(), end);
}

fn define_edge_to_infinity<T: CoordFloat + GeoFloat>(
    center_coord: &Coord<T>,
    triangle: &DelaunayTriangle<T>,
    circumcentre: &Coord<T>,
    shared_edges: &Vec<Line<T>>,
    clipping_mask: &Rect<T>,
) -> Result<Option<Line<T>>>
where
    f64: From<T>,
{
    let tri: Triangle<T> = triangle.clone().into();
    for edge in tri.to_lines().iter() {
        let flipped_edge = Line::new(edge.end, edge.start);

        if !shared_edges.contains(&edge) && !shared_edges.contains(&flipped_edge) {
            // Get the line that passes from the circumcentre and is perpendicular to
            // the edge without a 3rd vertex
            let line: Line<T> = get_perpendicular_line(edge)?;

            let slope = line.slope();

            // Get the width and height of the clipping mask to ensure intersection with
            // the lines of infinity. 2 x the width or height should be sufficient
            let width_factor = clipping_mask.width() + clipping_mask.width();
            let height_factor = clipping_mask.height() + clipping_mask.height();

            // Lines need to move away from the circumcentre and travel
            // towards infinity
            let end = if slope.is_infinite() {
                // It is a vertical line so we need to ensure it moves up or down correctly
                let end_y = if circumcentre.y < center_coord.y {
                    circumcentre.y - line.dy().abs() * height_factor
                } else {
                    circumcentre.y + line.dy().abs() * height_factor
                };
                coord! {x: circumcentre.x, y: end_y}
            } else {
                // A sloping or horizontal line
                let b = circumcentre.y - line.slope() * circumcentre.x;
                let end_x = if circumcentre.x < center_coord.x {
                    circumcentre.x - line.dx().abs() * width_factor
                } else {
                    circumcentre.x + line.dx().abs() * width_factor
                };
                let end_y = line.slope() * end_x + b;

                coord! {x: end_x, y: end_y}
            };

            return Ok(Some(Line::new(circumcentre.clone(), end)));
        }
    }
    Ok(None)
}

fn find_mean_vertex<T: CoordFloat>(vertices: &Vec<Coord<T>>) -> Result<Coord<T>> {
    let mut mean_vertex = coord! {
            x: T::from(0.).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?,
            y: T::from(0.).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?,
    };
    vertices.iter().for_each(|x| {
        mean_vertex.x = mean_vertex.x + x.x;
        mean_vertex.y = mean_vertex.y + x.y;
    });

    let num_vertices =
        T::from(vertices.len()).ok_or(VoronoiDiagramError::CannotConvertBetweenGeoGenerics)?;

    mean_vertex.x = mean_vertex.x / num_vertices;
    mean_vertex.y = mean_vertex.y / num_vertices;
    Ok(mean_vertex)
}

fn construct_edges_to_inf<T: CoordFloat + GeoFloat>(
    triangles: &Vec<DelaunayTriangle<T>>,
    vertices: &Vec<Coord<T>>,
    edges: &mut Vec<Line<T>>,
    neighbours: &Vec<Vec<Option<usize>>>,
    clipping_mask: &Polygon<T>,
) -> Result<()>
where
    f64: From<T>,
{
    // Find the mean vertex to determine the direction
    // of edges going to infinity.
    let mean_vertex: Coord<T> = find_mean_vertex(vertices)?;
    let clipping_bounds = clipping_mask
        .bounding_rect()
        .ok_or(VoronoiDiagramError::CannotDetermineBoundsFromClipppingMask)?;

    // Get the vertices with connections to infinity
    let mut infinity_vertices: Vec<usize> = Vec::new();
    for neighbour in neighbours.iter() {
        if neighbour.contains(&None) {
            // Unwrap here is ok as the filtered vec can only be [None, Some(idx)]
            let tri_idx = neighbour[1].unwrap();
            let inf_vertex = define_edge_to_infinity(
                &mean_vertex,
                &triangles[tri_idx],
                &vertices[tri_idx],
                &edges,
                &clipping_bounds,
            )?
            .ok_or(VoronoiDiagramError::CannotComputeExpectedInfinityVertex)?;

            // TODO: trim vertex at the intersection point with a line from the clipping_mask
            // then add the line to edges
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub enum VoronoiDiagramError {
    DelaunayError(DelaunayTriangulationError),
    CannotConvertBetweenGeoGenerics,
    CannotDetermineBoundsFromClipppingMask,
    CannotComputeExpectedInfinityVertex,
}

impl fmt::Display for VoronoiDiagramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoronoiDiagramError::DelaunayError(e) => {
                write!(f, "Delaunay Triangulation error: {}", e)
            }
            VoronoiDiagramError::CannotConvertBetweenGeoGenerics => {
                write!(f, "Cannot convert between Geo generic types")
            }
            VoronoiDiagramError::CannotDetermineBoundsFromClipppingMask => {
                write!(f, "Cannot determine the bounds from the clipping mask")
            }
            VoronoiDiagramError::CannotComputeExpectedInfinityVertex => {
                write(f, "Cannot compute expected boundary to infinity")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use geo_types::coord;

    use super::*;

    #[test]
    fn test_find_shared_edge() {
        let triangles: Vec<DelaunayTriangle<f64>> = vec![
            Triangle::new(
                coord! {x: 0., y: 0.},
                coord! {x: 1., y: 1.},
                coord! {x: 2., y: 0.},
            )
            .into(),
            Triangle::new(
                coord! {x: 1., y: 1.},
                coord! {x: 2., y: 0.},
                coord! {x: 3., y: 1.},
            )
            .into(),
        ];

        let (neighbours, shared_edges) = find_shared_edges(&triangles);

        assert_eq!(
            neighbours,
            vec![
                vec![Some(0), Some(1)],
                vec![None, Some(0)],
                vec![None, Some(1)]
            ]
        );

        assert_eq!(
            shared_edges,
            vec![Line::new(coord! { x: 1.0, y: 1.0}, coord! {x: 2.0, y: 0.}),]
        )
    }

    #[test]
    fn test_find_mean_vertex() {
        let vertices = vec![coord! {x: 1., y: 1.}, coord! {x: 2., y: 2.}];
        assert_eq!(find_mean_vertex(&vertices), coord! {x: 1.5, y:1.5});
    }

    #[test]
    fn test_get_perpendicular_line() {
        // Vertical line
        let line = Line::new(coord! {x: 0., y: 0.}, coord! {x: 0., y: 1.});
        assert_eq!(
            get_perpendicular_line(&line).unwrap(),
            Line::new(coord! {x: 0., y: 0.}, coord! {x: -1.0, y: 0.})
        );
        // Horizontal line
        let line = Line::new(coord! {x: 0., y: 0.}, coord! {x: 1., y: 0.});
        assert_eq!(
            get_perpendicular_line(&line).unwrap(),
            Line::new(coord! {x: 0., y: 0.}, coord! {x: 0., y: 1.})
        );

        // Check a diagonal line with a new starting point
        let line = Line::new(coord! {x: 0., y: 0.}, coord! {x: 2., y: 2.});
        assert_eq!(
            get_perpendicular_line(&line).unwrap(),
            Line::new(coord! {x: 1., y: 1.}, coord! {x: 3., y: -1.})
        );

        let line = Line::new(coord! {x: 0., y: 0.}, coord! {x: 2., y: -1.});
        assert_eq!(
            get_perpendicular_line(&line).unwrap(),
            Line::new(coord! {x: 1., y: -0.5}, coord! {x: 3., y: 3.5})
        );
    }

    #[test]
    fn test_define_edge_to_infinity() {
        let tri: DelaunayTriangle<_> = Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 0., y: 1.},
            coord! {x: 1., y: 1.},
        )
        .into();
        let tri2: DelaunayTriangle<_> = Triangle::new(
            coord! {x: 0., y: 0.},
            coord! {x: 0., y: 1.},
            coord! {x: -1., y: 2.},
        )
        .into();
        let tri3: DelaunayTriangle<_> = Triangle::new(
            coord! {x: 0., y: 1.},
            coord! {x: -1., y: 2.},
            coord! {x: 1., y: 1.},
        )
        .into();

        let bounds = Rect::new(coord! {x: -2., y: -2.}, coord! {x: 2., y: 2.});

        let circumcentre = tri.get_circumcircle().unwrap().center;
        let center_coord = coord! { x: 0., y: 1.};
        let shared_edges = vec![
            Line::new(coord! {x: 0., y: 0.}, coord! {x: 0., y: 1. }),
            Line::new(coord! {x: 0., y: 1.}, coord! {x: 1., y: 1. }),
            Line::new(coord! {x: 0., y: 1.}, coord! {x: -1., y: 2. }),
        ];
        let perpendicular_line =
            define_edge_to_infinity(&center_coord, &tri, &circumcentre, &shared_edges, &bounds)
                .unwrap();
        assert_eq!(
            perpendicular_line.unwrap(),
            Line::new(coord! {x: 0.5, y: 0.5}, coord! {x: 8.5, y: -7.5},)
        );

        let circumcentre = tri2.get_circumcircle().unwrap().center;
        let perpendicular_line =
            define_edge_to_infinity(&center_coord, &tri2, &circumcentre, &shared_edges, &bounds)
                .unwrap();
        assert_eq!(
            perpendicular_line.unwrap(),
            Line::new(coord! {x: -1.5, y: 0.5}, coord! {x: -9.5, y: -3.5},)
        );

        let circumcentre = tri3.get_circumcircle().unwrap().center;
        let perpendicular_line =
            define_edge_to_infinity(&center_coord, &tri3, &circumcentre, &shared_edges, &bounds)
                .unwrap();
        assert_eq!(
            perpendicular_line.unwrap(),
            Line::new(coord! {x: 0.5, y: 2.5}, coord! {x: 16.5, y: 34.5},)
        );
    }
}
