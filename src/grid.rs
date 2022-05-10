extern crate ndarray;
use std::rc::Rc;
use std::collections::HashMap;

/// The Grid struct represents the physical space over which the PDE is defined.
#[derive(Debug, PartialEq)]
pub struct Grid(ndarray::ArrayD<Point>);

impl Grid {
    pub(crate) fn indexed_iter(&self) -> ndarray::iter::IndexedIter<Point, ndarray::Dim<ndarray::IxDynImpl>> {
        self.0.indexed_iter()
    }

    pub(crate) fn iter(&self) -> ndarray::iter::Iter<'_, Point, ndarray::Dim<ndarray::IxDynImpl>> {
        self.0.iter()
    }
}

impl core::ops::Index<ndarray::Dim<ndarray::IxDynImpl>> for Grid {
    type Output = Point;

    fn index(self: &'_ Self, index: ndarray::Dim<ndarray::IxDynImpl>) -> &'_ Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<ndarray::Dim<ndarray::IxDynImpl>> for Grid {
    fn index_mut(self: &'_ mut Self, index: ndarray::Dim<ndarray::IxDynImpl>) -> &'_ mut Self::Output {
        &mut self.0[index]
    }
}

/// The ValVector struct simply stores a 1D array containing the quantity
/// of interest at each point on the grid.
#[derive(Clone, Debug, PartialEq)]
pub struct ValVector(pub(crate) ndarray::Array1<f64>);

impl ValVector {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        if self.len() == 0 { true }
        else { false }
    }

    fn vals(&self) -> &ndarray::Array1<f64> {
        &self.0
    }

    pub(crate) fn as_ndarray(&self) -> &ndarray::Array1<f64> {
        self.vals()
    }

    /// Returns a GridScalar instance that contains the values along the
    /// specified coordinate axis that correspond to the indices of the
    /// GridScalar. Use this if you need to add/multiply/etc. the value
    /// of interest by the axis coordinate (e.g., x*dT/dx, or y+T).
    /// # Arguments
    /// * `spec` - reference counted smart pointer to a GridSpec
    /// * `dimension` - the axis for which the values are desired
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::grid::{GridScalar, CartesianGridSpec, AxisSetup};
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let mut axs_init = HashMap::new();
    /// axs_init.insert('x', x_init);
    /// axs_init.insert('y', y_init);
    /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
    /// let temperature = GridScalar::zeros(Rc::clone(&spec));
    /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 'x');
    /// let y_vals = GridScalar::axis_vals(spec, 'y');
    /// let x_plus_temp = &x_vals + &temperature;
    /// let temp_minus_y = &temperature - &y_vals;
    /// assert_eq!(x_plus_temp, x_vals);
    /// assert_eq!(temp_minus_y, -&y_vals);
    /// ```
    pub fn axis_vals<S: GridSpec>(spec: &S, axis_label: char) -> Self {
        let mut n = 1;
        for (_, elm) in spec.gridshape() {
            n *= elm;
        }
        let mut axis: ndarray::Array1<f64> = ndarray::arr1(&vec![0.; n][..]);
        let _ = spec.grid().iter().map(|point| {
            axis[point.idx] = point.coord[&axis_label];
        }).collect::<()>();
        ValVector(axis)
    }
}

impl core::ops::Index<usize> for ValVector {
    type Output = f64;

    fn index(self: &'_ Self, index: usize) -> &'_ Self::Output {
        &self.0[index]
    }
}

impl core::ops::IndexMut<usize> for ValVector {
    fn index_mut(self: &'_ mut Self, index: usize) -> &'_ mut Self::Output {
        &mut self.0[index]
    }
}

pub struct GridSpace(HashMap<char, AxisSetup>);

impl GridSpace {
    pub fn new() -> Self {
        GridSpace(HashMap::new())
    }

    pub fn add_axis(&mut self, label: char, start: f64, delta: f64, steps: usize, periodic: bool) {
        let axis = AxisSetup::new(start, delta, steps, periodic);
        self.0.insert(label, axis);
    }
}

/// For consistency, this AxisSetup struct is used as an argument when
/// constructing GridSpecs. It contains the minimum axis value
/// (`start: f64`), the spacing of the axis points (`delta: f64`),
/// and the number of axis points including the start value
/// (`steps: usize`).
#[derive(Clone)]
struct AxisSetup {
    start: f64,
    delta: f64,
    steps: usize,
    periodic: bool,
}

impl AxisSetup {
    /// Returns an AxisSetup
    /// # Arguments
    /// * `start` - the minimum axis value
    /// * `delta` - the spacing btween axis points
    /// * `steps` - the number of axis points including start
    /// # Examples
    /// ```should_panic
    /// use rustencils::grid::AxisSetup;
    /// let ax = AxisSetup::new(0., 0., 100);
    /// ```
    /// 
    /// ```should_panic
    /// use rustencils::grid::AxisSetup;
    /// let ax = AxisSetup::new(0., 0.1, 0);
    /// ```
    /// 
    /// ```
    /// use rustencils::grid::AxisSetup;
    /// let ax = AxisSetup::new(0., 0.1, 100);
    /// ```
    fn new(start: f64, delta: f64, steps: usize, periodic: bool) -> Self {
        assert_ne!(delta, 0., "Delta cannot be zero!");
        assert_ne!(steps, 0, "Steps cannot be zero!");
        assert_ne!(steps, 1, "Steps cannot be one!");
        AxisSetup {
            delta: delta.abs(),
            start,
            steps,
            periodic,
        }
    }
}

/// The Point struct represents a single point on the Grid. Each Point
/// contains a vector of the axis values at that Point, as well as an
/// index that corresponds to the position within the GridQty that
/// represents the value of interest at that Point.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Point {
    pub(crate) coord: HashMap<char, f64>,
    pub(crate) idx: usize,
}

/// Since the physical space of the PDE can be defined in multiple
/// coordinate systems, the GridSpec trait is used to identify those
/// structs that can be used for this purpose. The trait defines the
/// necessary methods for a struct that specifies a certain type of
/// grid. For example, this trait is implemented by the
/// CartesianGridSpec type. It would also need to be implemented for
/// a SphericalGridSpec or PolarGridSpec.
pub trait GridSpec {
    /// Returns a Vector of characters that represent the axis labels
    fn axis_chars(&self) -> Vec<char>;
    /// Returns a shared reference to the HashMap containing the sets
    /// of points along the coordinate axes that make up the grid
    fn coords(&self) -> &HashMap<char, Vec<f64>>;
    /// Returns a usize that represents the dimensionality of the grid.
    fn ndim(&self) -> usize;
    /// Returns a shared reference to a HashMap that contains the number
    /// of points along each axis.
    fn gridshape(&self) -> &HashMap<char, usize>;
    /// Returns a shared reference to a HashMap that contains the axis
    /// indices for the grid.
    fn grid_axes(&self) -> &HashMap<char, usize>;
    /// Returns a shared reference to a HashMap that contains the spacing
    /// between the points on the coordinate axes.
    fn spacing(&self) -> &HashMap<char, f64>;
    /// Returns a shared reference to the Grid instance.
    fn grid(&self) -> &Grid;
    /// Returns a shared reference to a Vector of Points that represent the
    /// boundary points along the specified axis. Here, the boundary
    /// points refer to the outermost set of points along the edge of the
    /// grid.
    fn bound_pts(&self, axis: char, side: BoundarySide) -> &Vec<Point>;
    /// Returns a shared reference to a Hashmap that signifies whether
    /// each axis is periodic or not.
    fn periodic_axs(&self) -> &HashMap<char, bool>;
    /// Returns a shared reference to a vector of GridScalars.
    fn scalars(&self) -> &HashMap<char, GridScalar>;
    fn scalars_mut(&mut self) -> &mut HashMap<char, GridScalar>;
    /*
    /// Returns an owned vector of usizes that represent the index values
    /// of the boundary points along the specified axis. Here, the boundary
    /// points refer to the outermost set of points along the edge of the
    /// grid.
    fn bound_idxs(&self, dimension: usize, side: crate::boundaries::BoundarySide) -> Vec<usize>;
    */
}

pub enum BoundarySide {
    Low,
    High,
}

#[derive(Debug, PartialEq)]
struct BoundaryPoints {
    /// Points that comprise the boundary at the low end of the axis
    low: Vec<Point>,
    /// Points that comprise the boundary at the high end of the axis
    high: Vec<Point>,
}

/// The CartesianGridSpec struct represents the specifications of a
/// Grid in Cartesian coordinates. The dimensionality can be any size,
/// which means it could be more or less than the standard 3-dimensional
/// Cartesian coordinate space.
pub struct CartesianGridSpec {
    /// Vector of characters that represent the axis labels
    /// (e.g., ['x','y','z']).
    axis_chars: Vec<char>,
    /// HashMap containing the sets of points along the coordinate axes
    /// that make up the grid (e.g., { ('x', [x0,x1,x2,...,xm]),
    /// ('y', [y0,y1,y2,...,yn]) }).
    coords: HashMap<char, Vec<f64>>,
    /// A usize that represents the dimensionality of the grid.
    ndim: usize,
    /// HashMap that contains the number of points along each axis
    /// (e.g., { ('x', m), ('y', n) }).
    gridshape: HashMap<char, usize>,
    /// HashMap that contains the axis number of the grid associated with each label
    grid_axes: HashMap<char, usize>,
    /// HashMap that contains the spacing between the points on the
    /// coordinate axes (e.g., { ('x', (x1-x0)), ('y', (y1-y0)) }).
    spacing: HashMap<char, f64>,
    /// The Grid instance specified by this struct.
    grid: Grid,
    /// HashMap containing the Points that correspond to the Grid
    /// boundaries. Only non-periodic boundary points are included.
    /// Points that are part of more than one boundary, such
    /// as corners, are currently included and thus the behavior of 
    /// values at these points may be undefined.
    /// (e.g., { ('x', BoundaryPoints{...}), ('y', BoundaryPoints{...}) })
    boundary_pts: HashMap<char, BoundaryPoints>,
    /// HashMap that contains booleans signifying whether each axis
    /// is periodic.
    periodic_axs: HashMap<char, bool>,
    /// HashMap of GridScalars which are defined on the Grid
    scalars: HashMap<char, GridScalar>,
}

impl GridSpec for CartesianGridSpec {
    fn axis_chars(&self) -> Vec<char> { self.axis_chars.clone() }
    fn coords(&self) -> &HashMap<char, Vec<f64>> { &self.coords }
    fn ndim(&self) -> usize { self.ndim }
    fn gridshape(&self) -> &HashMap<char, usize> { &self.gridshape }
    fn grid_axes(&self) -> &HashMap<char, usize> { &self.grid_axes }
    fn spacing(&self) -> &HashMap<char, f64> { &self.spacing }
    fn grid(&self) -> &Grid { &self.grid }
    fn periodic_axs(&self) -> &HashMap<char, bool> { &self.periodic_axs }
    
    fn bound_pts(&self, axis: char, side: BoundarySide) -> &Vec<Point> {
        match side {
            BoundarySide::Low => &self.boundary_pts[&axis].low,
            BoundarySide::High => &self.boundary_pts[&axis].high,
        }
    }

    fn scalars(&self) -> &HashMap<char, GridScalar> { &self.scalars }
    fn scalars_mut(&mut self) -> &mut HashMap<char, GridScalar> { &mut self.scalars }

    /*
    fn bound_idxs(&self, dimension: usize, side: crate::boundaries::BoundarySide) -> Vec<usize> {
        let pts = self.bound_pts(dimension, side);
        let mut idxs = Vec::new();
        for pt in pts.iter() { idxs.push(pt.idx); }
        idxs
    }
    */
}

impl CartesianGridSpec {
    // Potential issue with casting usize to f64 if high precision required for
    // axis values.

    /// Returns a CartesianGridSpec built from a vector of AxisSetup structs.
    /// # Arguments
    /// * `axes` - vector containing one or more instances of AxisSetup
    /// # Examples
    /// ```
    /// use std::collections::HashMap;
    /// use rustencils::grid::{AxisSetup, GridSpec, CartesianGridSpec};
    /// let x = AxisSetup::new(0., 0.01, 100);
    /// let y = x.clone();
    /// let mut axs = HashMap::new();
    /// axs.insert('x', x);
    /// axs.insert('y', y);
    /// let spec = CartesianGridSpec::new(axs);
    /// assert_eq!(spec.gridshape()[&'x'], 100);
    /// assert_eq!(spec.gridshape()[&'y'], 100);
    /// assert_eq!(spec.spacing()[&'x'], 0.01);
    /// assert_eq!(spec.spacing()[&'y'], 0.01);
    /// ```
    pub fn new(axes: GridSpace) -> Self {
        let axis_chars: Vec<char> = axes.0.iter().map(|(label, _)| *label).collect();
        // grid_axes holds the axis index values that correspond to the axes of the grid object
        let grid_axes: HashMap<char, usize> = axis_chars.iter().enumerate().map(|(index, label)| (*label, index)).collect();
        // gridshape holds the number of steps for each axis
        let gridshape: HashMap<char, usize> = axes.0.iter().map(|(label, axis)| (*label, axis.steps)).collect();
        // calculate the full set of coordinates for each axis based on the
        // AxisSetup specifications
        let coords: HashMap<char, Vec<f64>> = axes.0.iter().map(|(label, axis)| {
                let mut set = Vec::with_capacity(axis.steps);
                for i in 0..axis.steps {set.push(axis.start + (i as f64)*axis.delta);}
                (*label, set)
            }).collect();
        // spacing holds the delta value for each axis
        let spacing: HashMap<char, f64> = axes.0.iter().map(|(label, axis)| (*label, axis.delta)).collect();
        // initialize the grid with default values
        let mut grid_init_vec: Vec<usize> = vec![0; axis_chars.len()];
        let _: () = grid_axes.iter().map(|(label, axnum)| {
            grid_init_vec[*axnum] = gridshape[label];
        }).collect();
        let mut grid: ndarray::ArrayD<Point> = ndarray::Array::default(grid_init_vec);
        let mut count = 0;
        // popluate the grid with Point structs based on coordinates
        let _: () = grid.indexed_iter_mut().map(|(indices,pt)| {
            pt.coord = HashMap::new();
            for ax in axis_chars.iter() {
                pt.coord.insert(*ax, coords[ax][indices[grid_axes[ax]]]);
            }
            pt.idx = count;
            count += 1;
        }).collect();
        let periodic_axs: HashMap<char, bool> = axes.0.iter().map(|(label, axis)| (*label, axis.periodic)).collect();
        let mut boundary_pts: HashMap<char, BoundaryPoints> = HashMap::new();
        // populate the boundary points
        for (ax, periodic) in periodic_axs.iter() {
            if !periodic {
                let mut low_pts: Vec<Point> = Vec::new();
                let mut high_pts: Vec<Point> = Vec::new();
                let low_slc = grid.slice_axis(ndarray::Axis(grid_axes[ax]), ndarray::Slice::from(0..1));
                let high_slc = grid.slice_axis(ndarray::Axis(grid_axes[ax]), ndarray::Slice::from(-1..-2));
                // TODO: Determine a way to better handle corners and edges where
                // boundaries meet.
                for pt in low_slc.iter() { low_pts.push(pt.clone()); }
                for pt in high_slc.iter() { high_pts.push(pt.clone()); }
                boundary_pts.insert(*ax, BoundaryPoints{low: low_pts, high: high_pts});
            }
        }

        CartesianGridSpec {
            axis_chars,
            spacing,
            gridshape,
            grid_axes,
            ndim: axes.0.len(),
            coords,
            grid: Grid(grid),
            boundary_pts,
            periodic_axs,
            scalars: HashMap::new(),
        }
    }

    pub fn scalars(&self) -> &HashMap<char, GridScalar> { &self.scalars }

    pub fn scalars_mut(&mut self) -> &mut HashMap<char, GridScalar> { &mut self.scalars }
}

// /// The GridQty trait is meant to leave open the possibility of in the
// /// future adding something like a GridVector struct that would store
// /// a vector value for each point on the grid as opposed to the scalar
// /// values stored by the GridScalar struct. However, it is more likely
// /// that this trait may be deprecated in the future and a GridVector
// /// struct would just contain a vector of GridScalars.
// pub trait GridQty<S> where S: GridSpec {
//     /// Returns an Rc pointing to the GridSpec held by the GridQty.
//     fn spec(&self) -> Rc<S>;
//     /// Returns a shared reference to the ValVector held by the GridQty.
//     fn gridvals(&self) -> &ValVector;
//     /// Returns a mutable reference to the ValVector held by the GridQty.
//     fn gridvals_mut(&mut self) -> &mut ValVector;
//     /// Returns a shared reference to the Grid struct on which the
//     /// GridQty is defined.
//     fn grid(&self) -> &Grid;
//     /// A public API that allows the creation of a new GridQty simply
//     /// from its component parts (i.e., a GridSpec and a ValVector).
//     /// Because ValVector has a very limited public API, this method
//     /// is generally used to construct a new GridQty after performing
//     /// some operation on an existing GridQty.
//     /// (See rustencils::operator::OperatorMatrix::of)
//     /// # Arguments
//     /// * `spec` - reference counted smart pointer to a GridSpec
//     /// * `gridvals` - ValVector containing the quantity of interest
//     fn new(spec: Rc<S>, gridvals: ValVector) -> Self;
// }

/// The GridScalar struct is the type that represents the values of
/// interest. It contains a GridSpec and a ValVector.
// #[derive(Clone, Debug, PartialEq)]
pub struct GridScalar {
    /// A vector that just contains the values of interest at every point
    gridvals: ValVector,
    /// The boundary conditions that apply to the gridvals
    bound_conds: Vec<Box<dyn BoundaryCondition>>,
}

// impl<S> GridQty<S> for GridScalar<S> where S: GridSpec {
//     fn spec(&self) -> Rc<S> { Rc::clone(&self.spec) }
//     fn gridvals(&self) -> &ValVector { &self.gridvals }
//     fn gridvals_mut(&mut self) -> &mut ValVector { &mut self.gridvals }
//     fn grid(&self) -> &Grid { self.spec.grid() }
//     fn new(spec: Rc<S>, gridvals: ValVector) -> Self {
//         Self::new(spec, gridvals)
//     }
// }

impl GridScalar {
    /// Returns a shared reference to the ValVector held by the GridQty.
    pub fn gridvals(&self) -> &ValVector { &self.gridvals }
    /// Returns a mutable reference to the ValVector held by the GridQty.
    pub fn gridvals_mut(&mut self) -> &mut ValVector { &mut self.gridvals }

    pub fn add_bound_condition<B: BoundaryCondition>(&mut self, condition: B) {
        self.bound_conds.append(vec![Box::new(condition)]);
    }
    /// Private constructor
    pub(crate) fn new(gridvals: ValVector, bound_conds: Vec<Box<dyn BoundaryCondition>>) -> Self {
        GridScalar {
            gridvals,
            bound_conds,
        }
    }

    /// Returns a GridScalar where the value of interest at each point is
    /// equal to the value passed in as an argument
    /// # Arguments
    /// * `spec` - reference counted smart pointer to a GridSpec
    /// * `value` - the value that will be set at each grid point
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::grid::{GridScalar, CartesianGridSpec, AxisSetup};
    /// let x = AxisSetup::new(0., 0.01, 100);
    /// let y = x.clone();
    /// let mut axs = HashMap::new();
    /// axs.insert('x', x);
    /// axs.insert('y', y);
    /// let spec = Rc::new(CartesianGridSpec::new(axs));
    /// let temperature = GridScalar::uniform(spec, 0.5);
    /// assert_eq!(temperature.gridvals()[0], 0.5);
    /// assert_eq!(temperature.gridvals().len(), 100*100);
    /// ```
    pub fn uniform<S: GridSpec>(spec: &mut S, label: char, value: f64) {
        let mut n = 1;
        for (_, elm) in spec.gridshape() {
            n *= elm;
        }
        let gridvals: ndarray::Array1<f64> = ndarray::arr1(&vec![value; n][..]);
        spec.scalars_mut().insert(
            label,
            GridScalar {
                gridvals: ValVector(gridvals),
                bound_conds: Vec::new(),
            }
        );
    }

    /// Returns a GridScalar where the value of interest at each point is
    /// equal to one.
    /// # Arguments
    /// * `spec` - reference counted smart pointer to a GridSpec
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::grid::{GridScalar, CartesianGridSpec, AxisSetup};
    /// let x = AxisSetup::new(0., 0.01, 100);
    /// let y = x.clone();
    /// let mut axs = HashMap::new();
    /// axs.insert('x', x);
    /// axs.insert('y', y);
    /// let spec = Rc::new(CartesianGridSpec::new(axs));
    /// let temperature = GridScalar::ones(spec);
    /// assert_eq!(temperature.gridvals()[0], 1.);
    /// assert_eq!(temperature.gridvals().len(), 100*100);
    /// ```
    pub fn ones<S: GridSpec>(spec: &mut S, label: char) {
        GridScalar::uniform(spec, label, 1.);
    }

    /// Returns a GridScalar where the value of interest at each point is
    /// equal to zero.
    /// # Arguments
    /// * `spec` - reference counted smart pointer to a GridSpec
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::grid::{GridScalar, CartesianGridSpec, AxisSetup};
    /// let x = AxisSetup::new(0., 0.01, 100);
    /// let y = x.clone();
    /// let mut axs = HashMap::new();
    /// axs.insert('x', x);
    /// axs.insert('y', y);
    /// let spec = Rc::new(CartesianGridSpec::new(axs));
    /// let temperature = GridScalar::zeros(spec);
    /// assert_eq!(temperature.gridvals()[0], 0.);
    /// assert_eq!(temperature.gridvals().len(), 100*100);
    /// ```
    pub fn zeros<S: GridSpec>(spec: &mut S, label: char) {
        GridScalar::uniform(spec, label, 0.);
    }
}

impl<'a, 'b> std::ops::Sub<&'b ValVector> for &'a ValVector {
    type Output = ValVector;

    fn sub(self, other: &'b ValVector) -> Self::Output {
        if self.len() == other.len() {
            ValVector(self.0 - other.0)
        }
        else { panic!("Error subtracting ValVectors! Ensure sizes are the same.") }
    }
}

impl<'a> std::ops::Sub<f64> for &'a ValVector {
    type Output = ValVector;

    fn sub(self, other: f64) -> Self::Output {
        ValVector(self.0 - other)
    }
}

impl<'a> std::ops::Sub<&'a ValVector> for f64 {
    type Output = ValVector;

    fn sub(self, other: &'a ValVector) -> Self::Output {
        ValVector(self - other.0)
    }
}

impl<'a, 'b> std::ops::Add<&'b ValVector> for &'a ValVector {
    type Output = ValVector;

    fn add(self, other: &'b ValVector) -> Self::Output {
        if self.len() == other.len() {
            ValVector(self.0 + other.0)
        }
        else { panic!("Error adding ValVectors! Ensure sizes are the same.") }
    }
}

impl<'a> std::ops::Add<f64> for &'a ValVector {
    type Output = ValVector;

    fn add(self, other: f64) -> Self::Output {
        ValVector(self.0 + other)
    }
}

impl<'a> std::ops::Add<&'a ValVector> for f64 {
    type Output = ValVector;

    fn add(self, other: &'a ValVector) -> Self::Output {
        ValVector(self + other.0)
    }
}

impl<'a, 'b> std::ops::Mul<&'b ValVector> for &'a ValVector {
    type Output = ValVector;

    fn mul(self, other: &'b ValVector) -> Self::Output {
        if self.len() == other.len() {
            ValVector(self.0 * other.0)
        }
        else { panic!("Error multiplying ValVectors! Ensure sizes are the same.") }
    }
}

impl<'a> std::ops::Mul<f64> for &'a ValVector {
    type Output = ValVector;

    fn mul(self, other: f64) -> Self::Output {
        ValVector(self.0 * other)
    }
}

impl<'a> std::ops::Mul<&'a ValVector> for f64 {
    type Output = ValVector;

    fn mul(self, other: &'a ValVector) -> Self::Output {
        ValVector(self * other.0)
    }
}

impl<'a, 'b> std::ops::Div<&'b ValVector> for &'a ValVector {
    type Output = ValVector;

    fn div(self, other: &'b ValVector) -> Self::Output {
        if self.len() == other.len() {
            ValVector(self.0 / other.0)
        }
        else { panic!("Error dividing ValVectors! Ensure sizes are the same.") }
    }
}

impl<'a> std::ops::Div<f64> for &'a ValVector {
    type Output = ValVector;

    fn div(self, other: f64) -> Self::Output {
        ValVector(self.0 / other)
    }
}

impl<'a> std::ops::Div<&'a ValVector> for f64 {
    type Output = ValVector;

    fn div(self, other: &'a ValVector) -> Self::Output {
        ValVector(self / other.0)
    }
}

impl<'a> std::ops::Neg for &'a ValVector {
    type Output = ValVector;

    fn neg(self) -> Self::Output {
        ValVector(-self.0)
    }
}

/// The BoundaryHandler struct simply holds a vector of BoundaryConditions
/// and is responsible for setting the boundary points to the appropriate
/// values.
struct BoundaryHandler {
    conditions: Vec<Box<dyn BoundaryCondition>>,
}

impl BoundaryHandler {
    /// Constructs a new BoundaryHandler instance. The caller is responsible
    /// for ensuring that there is only one condition per combination of
    /// axis and side. Currently, repeats are ignored in
    /// BoundaryHandler::set_bound_vals().
    /// # Arguments
    /// * `conditions` - Vector of BoundaryConditions for the current system
    fn new() -> Self {
        BoundaryHandler {
            conditions: Vec::new(),
        }
    }

    /// Sets the boundary points to the appropriate values by iterating
    /// through the boundary conditions.
    /// # Arguments
    /// * `time` - The value of the current timestep
    /// * `qty` - Exclusive reference to the GridQty that will have its values modified
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::grid::{GridScalar, CartesianGridSpec, AxisSetup};
    /// use rustencils::grid::{BoundarySide, BoundaryHandler, BoundaryCondition, DirichletConstant};
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let mut axs_init = HashMap::new();
    /// axs_init.insert('x', x_init);
    /// axs_init.insert('y', y_init);
    /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
    /// let dir_bc = Box::new(DirichletConstant::new(1., &*spec, 'x', BoundarySide::Low));
    /// let mut bcs = BoundaryHandler::new(vec![dir_bc]);
    /// let mut temperature = GridScalar::zeros(spec);
    /// bcs.set_bound_vals(0., &mut temperature);
    /// ```
    /// TODO: Improve testing of this method.
    /// Note: Must update constructor if handling of repeats changes!
    fn set_bound_vals(&mut self, time: f64, qty: &mut GridScalar) {
        let mut settable: Vec<(usize, f64)> = Vec::new();
        for cndtn in self.conditions.iter_mut() {
            settable.append(&mut cndtn.generate_vals(time));
        }
        let gridvals = qty.gridvals_mut();
        let mut used_indices: Vec<usize> = Vec::new();
        for (idx, val) in settable.iter() {
            if used_indices.contains(idx) {
                continue
            }
            gridvals[*idx] = *val;
            used_indices.push(*idx);
        }
    } // maybe call this set_BCs?
    // fn check_bc_type(&self) -> String;
}

pub trait BoundaryCondition {
    fn generate_vals(&mut self, time: f64) -> Vec<(usize, f64)>;
}

/// The DirichletConstant is a BoundaryCondition that is not dependent on
/// space or time.
pub struct DirichletConstant {
    /// bc_list is a Vec that holds tuples containing the index of the
    /// crate::grid::ValVector and the value to be set at that index.
    bc_list: Vec<(usize, f64)>,
}

impl BoundaryCondition for DirichletConstant {
    fn generate_vals(&mut self, _time: f64) -> Vec<(usize, f64)> { self.bc_list.clone() }
}

impl DirichletConstant {
    /// Returns a new DirichletConstant instance based on the given value,
    /// GridSpec, axis, and side arguments.
    /// # Arguments
    /// * `value` - The value to be set at the boundary
    /// * `spec` - Reference to a GridSpec object
    /// * `axis` - The character that labels which axis will have its boundary set
    /// * `side` - The side of the given axis to be set to the given value
    pub fn new<S: GridSpec>(value: f64, spec: &S, axis: char, side: BoundarySide) -> Self {
        let mut bc_list = Vec::new();
        for pt in spec.bound_pts(axis, side).iter() {
            bc_list.push((pt.idx, value));
        }

        DirichletConstant {
            bc_list,
        }
    }
}

/// The DirichletFunction is a boundary condition that may depend on space and time
/// through a function pointer held by the object.
pub struct DirichletFunction<F> {
    // fnctn: Box<dyn Fn(f64, &Point) -> f64>,
    fnctn: F,
    pts: Vec<Point>,
}

impl<F: Fn(f64, &Point)->f64> BoundaryCondition for DirichletFunction<F> {
    fn generate_vals(&mut self, time: f64) -> Vec<(usize, f64)> {
        let mut output = Vec::new();
        for pt in self.pts.iter() {
            output.push((pt.idx, (self.fnctn)(time, pt)));
        }

        output
    }
}

impl<F: Fn(f64, &Point)->f64> DirichletFunction<F> {
    /// Returns a new DirichletFunction instance based on the given function,
    /// GridSpec, axis, and side arguments.
    /// # Arguments
    /// * `fnctn` - The function that evaluates to the value at the boundary
    /// * `spec` - Reference to a GridSpec object
    /// * `axis` - The character that labels which axis will have its boundary set
    /// * `side` - The side of the given axis to be set with the given function
    pub fn new<S>(
        fnctn: F, // Box<dyn Fn(f64, &Point) -> f64>, 
        spec: &S, 
        axis: char, 
        side: BoundarySide
    ) -> Self 
        where S: GridSpec {
        DirichletFunction {
            fnctn,
            pts: spec.bound_pts(axis, side).clone(),
        }
    }
}

pub struct DirichletConstantVector {
    prev_val: Option<Vec<(usize, f64)>>,
    remaining: Vec<f64>,
    pts: Vec<Point>,
}

impl BoundaryCondition for DirichletConstantVector {
    fn generate_vals(&mut self, _time: f64) -> Vec<(usize, f64)> {
        if self.remaining.is_empty() {
            if let Some(prev) = &self.prev_val {
                return prev.clone()
            }
        }
        let val = self.remaining.remove(0);
        let mut output = Vec::new();
        for pt in self.pts.iter() {
            output.push((pt.idx, val))
        }
        if self.remaining.is_empty() {
            self.prev_val = Some(output.clone())
        }

        output
    }
}

impl DirichletConstantVector {
    /// Returns a new DirichletConstantVector instance based on the given values,
    /// GridSpec, axis, and side arguments.
    /// # Arguments
    /// * `values` - The vector of value to be set at the boundary
    /// * `spec` - Reference to a GridSpec object
    /// * `axis` - The character that labels which axis will have its boundary set
    /// * `side` - The side of the given axis to be set to the given value
    pub fn new<S: GridSpec>(values: Vec<f64>, spec: &S, axis: char, side: BoundarySide) -> Self {
        DirichletConstantVector {
            prev_val: None,
            remaining: values,
            pts: spec.bound_pts(axis, side).clone(),
        }
    }

    /// Returns a new DirichletConstantVector instance based on the given time,
    /// function, GridSpec, axis, and side arguments.
    /// # Arguments
    /// * `tstart` - The first time point of the
    /// * `timestep` - The amount by which to increment the time
    /// * `nsteps` - The number of time steps
    /// * `fnctn` - The function that evaluates to the boundary values
    /// * `spec` - Reference to a GridSpec object
    /// * `axis` - The character that labels which axis will have its boundary set
    /// * `side` - The side of the given axis to be set to the given value
    pub fn from_function<S, F>(
        tstart: f64, 
        timestep: f64, 
        nsteps: usize, 
        fnctn: F, 
        spec: &S, 
        axis: char, 
        side: BoundarySide
    ) -> Self
        where
            S: GridSpec,
            F: Fn(f64) -> f64 {
        let mut values = Vec::new();
        for i in 0..nsteps {
            values.push(fnctn(tstart + timestep*(i as f64)));
        }

        DirichletConstantVector {
            prev_val: None,
            remaining: values,
            pts: spec.bound_pts(axis, side).clone()
        }
    }
}

/*
// To be implemented at later date:
pub struct DirichletFunctionVector {
    prev_fnctn: Option<fn(time: f64) -> f64>,
    remaining: Option<Vec<fn(time: f64) -> f64>>,
    pts: Vec<crate::grid::Point>,
}
*/