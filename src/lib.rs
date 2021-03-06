#![crate_name = "rustencils"]
pub mod grid {

    extern crate ndarray;

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
    
    /// For consistency, this AxisSetup struct is used as an argument when
    /// constructing GridSpecs. It contains the minimum axis value
    /// (`start: f64`), the spacing of the axis points (`delta: f64`),
    /// and the number of axis points including the start value
    /// (`steps: usize`).
    #[derive(Clone)]
    pub struct AxisSetup {
        start: f64,
        delta: f64,
        steps: usize,
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
        pub fn new(start: f64, delta: f64, steps: usize) -> Self {
            assert_ne!(delta, 0., "Delta cannot be zero!");
            assert_ne!(steps, 0, "Steps cannot be zero!");
            assert_ne!(steps, 1, "Steps cannot be one!");
            AxisSetup {
                delta: delta.abs(),
                start,
                steps,
            }
        }
    }

    /// The Point struct represents a single point on the Grid. Each Point
    /// contains a vector of the axis values at that Point, as well as an
    /// index that corresponds to the position within the GridQty that
    /// represents the value of interest at that Point.
    #[derive(Default, Clone, Debug, PartialEq)]
    pub struct Point {
        pub(crate) coord: Vec<f64>,
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
        /// Returns a shared reference to the vector containing the sets
        /// of points along the coordinate axes that make up the grid
        /// (e.g., [[x0,x1,x2,...,xm],[y0,y1,y2,...,yn]]).
        fn get_coords(&self) -> &Vec<Vec<f64>>;
        /// Returns a usize that represents the dimensionality of the grid.
        fn get_ndim(&self) -> usize;
        /// Returns a shared reference to a vector that contains the number
        /// of points along each axis (e.g., [m,n]).
        fn get_gridshape(&self) -> &Vec<usize>;
        /// Returns a shared reference to a vector that contains the spacing
        /// between the points on the coordinate axes (e.g., [(x1-x0),(y1-y0)]).
        fn get_spacing(&self) -> &Vec<f64>;
        /// Returns a shared reference to the Grid instance.
        fn get_grid(&self) -> &Grid;
        /// Returns an owned vector of usizes that represent the index values
        /// of the boundary points along the specified axis. Here, the boundary
        /// points refer to the outermost set of points along the edge of the
        /// grid.
        fn get_bound_idxs(&self, dimension: usize, side: crate::boundaries::BoundarySide) -> Vec<usize>;
        /// Returns an owned vector of Point structs that represent the
        /// boundary points along the specified axis. Here, the boundary
        /// points refer to the outermost set of points along the edge of the
        /// grid.
        fn get_bound_pts(&self, dimesnion: usize, side: crate::boundaries::BoundarySide) -> Vec<Point>;
    }

    /// The CartesianGridSpec struct represents the specifications of a
    /// Grid in Cartesian coordinates. The dimensionality can be any size,
    /// which means it could be more or less than the standard 3-dimensional
    /// Cartesian coordinate space.
    #[derive(Debug, PartialEq)]
    pub struct CartesianGridSpec {
        /// Vector containing the sets of points along the coordinate axes
        /// that make up the grid (e.g., [[x0,x1,x2,...,xm],[y0,y1,y2,...,yn]]).
        coords: Vec<Vec<f64>>,
        /// A usize that represents the dimensionality of the grid.
        ndim: usize,
        /// Vector that contains the number of points along each axis
        /// (e.g., [m,n]).
        gridshape: Vec<usize>,
        /// Vector that contains the spacing between the points on the
        /// coordinate axes (e.g., [(x1-x0),(y1-y0)]).
        spacing: Vec<f64>,
        /// The Grid instance specified by this struct.
        grid: Grid,
        /// Vector containing the Points that correspond to the Grid
        /// boundaries. The vector contains 2-element arrays. Each
        /// array corresponds to a different axis, and each element
        /// of the array corresponds to either the Low or High side
        /// of that axis.
        boundary_pts: Vec<[Vec<Point>;2]>,
    }

    impl GridSpec for CartesianGridSpec {
        fn get_coords(&self) -> &Vec<Vec<f64>> { &self.coords }
        fn get_ndim(&self) -> usize { self.ndim }
        fn get_gridshape(&self) -> &Vec<usize> { &self.gridshape }
        fn get_spacing(&self) -> &Vec<f64> { &self.spacing }
        fn get_grid(&self) -> &Grid { &self.grid }
        
        fn get_bound_pts(&self, dimension: usize, side: crate::boundaries::BoundarySide) -> Vec<Point> {
            let side_idx = match side {
                crate::boundaries::BoundarySide::Low => 0,
                crate::boundaries::BoundarySide::High => 1,
            };
            self.boundary_pts[dimension][side_idx].clone()
        }

        fn get_bound_idxs(&self, dimension: usize, side: crate::boundaries::BoundarySide) -> Vec<usize> {
            let pts = self.get_bound_pts(dimension, side);
            let mut idxs = Vec::new();
            for pt in pts.iter() { idxs.push(pt.idx); }
            idxs
        }
    }

    impl CartesianGridSpec {
        // Potential issue with casting usize to f64 if high precision required for
        // axis values.

        /// Returns a CartesianGridSpec built from a vector of AxisSetup structs.
        /// # Arguments
        /// * `axes` - vector containing one or more instances of AxisSetup
        /// # Examples
        /// ```
        /// use rustencils::grid::{AxisSetup, GridSpec, CartesianGridSpec};
        /// let x = AxisSetup::new(0., 0.01, 100);
        /// let y = x.clone();
        /// let axs = vec![x, y];
        /// let spec = CartesianGridSpec::new(axs);
        /// assert_eq!(spec.get_gridshape(), &vec![100,100]);
        /// assert_eq!(spec.get_spacing(), &vec![0.01,0.01]);
        /// ```
        pub fn new(axes: Vec<AxisSetup>) -> Self {
            // gridshape holds the number of steps for each axis
            let gridshape: Vec<usize> = axes.iter().map(|ax| ax.steps).collect();
            // calculate the full set of coordinates for each axis based on the
            // AxisSetup specifications
            let coords: Vec<Vec<f64>> = axes.iter().map(|ax| {
                    let mut set = Vec::with_capacity(ax.steps);
                    for i in 0..ax.steps {set.push(ax.start + (i as f64)*ax.delta);}
                    set
                }).collect();
            // spacing holds the delta value for each axis
            let spacing: Vec<f64> = axes.iter().map(|ax| ax.delta).collect();
            // initialize the grid with default values
            let mut grid: ndarray::ArrayD<Point> = ndarray::Array::default(gridshape.clone());
            let mut count = 0;
            // popluate the grid with Point structs based on coordinates
            let _ = grid.indexed_iter_mut().map(|(indices,pt)| {
                pt.coord = Vec::new();
                for i in 0..coords.len() {
                    pt.coord.push(coords[i][indices[i]]);
                }
                pt.idx = count;
                count += 1;
            }).collect::<()>();
            let mut boundary_pts: Vec<[Vec<Point>;2]> = Vec::new();
            // populate the boundary points vector
            for i in 0..axes.len() {
                boundary_pts.push([Vec::new(), Vec::new()]);
                for j in 0..2 {
                    let slc = match j {
                        0 => grid.slice_axis(ndarray::Axis(i), ndarray::Slice::from(0..1)),
                        1 => grid.slice_axis(ndarray::Axis(i), ndarray::Slice::from(-1..-2)),
                        _ => panic!("Error while constructing grid boundary points!"),
                    };

                    for pt in slc.iter() {
                        boundary_pts[i][j].push(pt.clone());
                    }
                }
            }
            CartesianGridSpec {
                spacing,
                gridshape,
                ndim: axes.len(),
                coords,
                grid: Grid(grid),
                boundary_pts,
            }
        }
    }

    /// The GridQty trait is meant to leave open the possibility of in the
    /// future adding something like a GridVector struct that would store
    /// a vector value for each point on the grid as opposed to the scalar
    /// values stored by the GridScalar struct. However, it is more likely
    /// that this trait may be deprecated in the future and a GridVector
    /// struct would just contain a vector of GridScalars.
    pub trait GridQty<S> where S: GridSpec {
        /// Returns an Rc pointing to the GridSpec held by the GridQty.
        fn get_spec(&self) -> Rc<S>;
        /// Returns a shared reference to the ValVector held by the GridQty.
        fn get_gridvals(&self) -> &ValVector;
        /// Returns a shared reference to the Grid struct on which the
        /// GridQty is defined.
        fn get_grid(&self) -> &Grid;
        /// A public API that allows the creation of a new GridQty simply
        /// from its component parts (i.e., a GridSpec and a ValVector).
        /// Because ValVector has a very limited public API, this method
        /// is generally used to construct a new GridQty after performing
        /// some operation on an existing GridQty.
        /// (See rustencils::operator::OperatorMatrix::of)
        /// # Arguments
        /// * `spec` - reference counted smart pointer to a GridSpec
        /// * `gridvals` - ValVector containing the quantity of interest
        fn new(spec: Rc<S>, gridvals: ValVector) -> Self;
    }

    use std::rc::Rc;
    
    /// The GridScalar struct is the type that represents the values of
    /// interest. It contains a GridSpec and a ValVector.
    #[derive(Clone, Debug, PartialEq)]
    pub struct GridScalar<S> {
        /// A reference counted smart pointer to a GridSpec
        spec: Rc<S>,
        /// A vector that just contains the values of interest at every point
        gridvals: ValVector, 
    }

    impl<S> GridQty<S> for GridScalar<S> where S: GridSpec {
        fn get_spec(&self) -> Rc<S> { Rc::clone(&self.spec) }
        fn get_gridvals(&self) -> &ValVector { &self.gridvals }
        fn get_grid(&self) -> &Grid { self.spec.get_grid() }
        fn new(spec: Rc<S>, gridvals: ValVector) -> Self {
            Self::new(spec, gridvals)
        }
    }

    impl<S> GridScalar<S> where S: GridSpec {
        /// Private constructor that is used by the implementation of
        /// GridQty::new().
        fn new(spec: Rc<S>, gridvals: ValVector) -> Self {
            GridScalar {
                spec,
                gridvals,
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
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// let x = AxisSetup::new(0., 0.01, 100);
        /// let y = x.clone();
        /// let axs = vec![x, y];
        /// let spec = Rc::new(CartesianGridSpec::new(axs));
        /// let temperature = GridScalar::uniform(spec, 0.5);
        /// assert_eq!(temperature.get_gridvals()[0], 0.5);
        /// assert_eq!(temperature.get_gridvals().len(), 100*100);
        /// ```
        pub fn uniform(spec: Rc<S>, value: f64) -> Self {
            let mut n = 1;
            for elm in spec.get_gridshape() {
                n *= elm;
            }
            let gridvals: ndarray::Array1<f64> = ndarray::arr1(&vec![value; n][..]);
            GridScalar{
                spec,
                gridvals: ValVector(gridvals),
            }
        }

        /// Returns a GridScalar where the value of interest at each point is
        /// equal to one.
        /// # Arguments
        /// * `spec` - reference counted smart pointer to a GridSpec
        /// # Examples
        /// ```
        /// use std::rc::Rc;
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// let x = AxisSetup::new(0., 0.01, 100);
        /// let y = x.clone();
        /// let axs = vec![x, y];
        /// let spec = Rc::new(CartesianGridSpec::new(axs));
        /// let temperature = GridScalar::ones(spec);
        /// assert_eq!(temperature.get_gridvals()[0], 1.);
        /// assert_eq!(temperature.get_gridvals().len(), 100*100);
        /// ```
        pub fn ones(spec: Rc<S>) -> Self {
            GridScalar::uniform(spec, 1.)
        }

        /// Returns a GridScalar where the value of interest at each point is
        /// equal to zero.
        /// # Arguments
        /// * `spec` - reference counted smart pointer to a GridSpec
        /// # Examples
        /// ```
        /// use std::rc::Rc;
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// let x = AxisSetup::new(0., 0.01, 100);
        /// let y = x.clone();
        /// let axs = vec![x, y];
        /// let spec = Rc::new(CartesianGridSpec::new(axs));
        /// let temperature = GridScalar::zeros(spec);
        /// assert_eq!(temperature.get_gridvals()[0], 0.);
        /// assert_eq!(temperature.get_gridvals().len(), 100*100);
        /// ```
        pub fn zeros(spec: Rc<S>) -> Self {
            GridScalar::uniform(spec, 0.)
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
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// let x_init = AxisSetup::new(0., 0.01, 100);
        /// let y_init = x_init.clone();
        /// let axs_init = vec![x_init, y_init];
        /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
        /// let temperature = GridScalar::zeros(Rc::clone(&spec));
        /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 0);
        /// let y_vals = GridScalar::axis_vals(spec, 1);
        /// let x_plus_temp = &x_vals + &temperature;
        /// let temp_minus_y = &temperature - &y_vals;
        /// assert_eq!(x_plus_temp, x_vals);
        /// assert_eq!(temp_minus_y, -&y_vals);
        /// ```
        pub fn axis_vals(spec: Rc<S>, dimension: usize) -> Self {
            let mut axis = GridScalar::zeros(Rc::clone(&spec));
            let _ = spec.get_grid().iter().map(|point| {
                axis.gridvals[point.idx] = point.coord[dimension];
            }).collect::<()>();
            axis
        }
    }

    impl<'a, 'b, S> std::ops::Sub<&'b GridScalar<S>> for &'a GridScalar<S> where S: GridSpec {
        type Output = GridScalar<S>;

        fn sub(self, other: &'b GridScalar<S>) -> Self::Output {
            if self.gridvals.len() == other.gridvals.len() &&
            Rc::ptr_eq(&self.get_spec(), &other.get_spec()) {
                let result = self.gridvals.vals() - other.gridvals.vals();
                GridScalar {
                    spec: Rc::clone(&self.spec),
                    gridvals: ValVector(result),
                }
            }
            else { panic!("Error subtracting GridScalars! Ensure sizes and GridSpecs are the same.") }
        }
    }

    impl<'a, S> std::ops::Sub<f64> for &'a GridScalar<S> {
        type Output = GridScalar<S>;

        fn sub(self, other: f64) -> Self::Output {
            let result  = self.gridvals.vals() - other;
            
            GridScalar {
                spec: Rc::clone(&self.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, S> std::ops::Sub<&'a GridScalar<S>> for f64 {
        type Output = GridScalar<S>;

        fn sub(self, other: &'a GridScalar<S>) -> Self::Output {
            let result = self - other.gridvals.vals();
            
            GridScalar {
                spec: Rc::clone(&other.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, 'b, S> std::ops::Add<&'b GridScalar<S>> for &'a GridScalar<S> where S: GridSpec {
        type Output = GridScalar<S>;

        fn add(self, other: &'b GridScalar<S>) -> Self::Output {
            if self.gridvals.len() == other.gridvals.len() &&
            Rc::ptr_eq(&self.get_spec(), &other.get_spec()) {
                let result = self.gridvals.vals() + other.gridvals.vals();
                GridScalar {
                    spec: Rc::clone(&self.spec),
                    gridvals: ValVector(result),
                }
            }
            else { panic!("Error adding GridScalars! Ensure sizes and GridSpecs are the same.") }
        }
    }

    impl<'a, S> std::ops::Add<f64> for &'a GridScalar<S> {
        type Output = GridScalar<S>;

        fn add(self, other: f64) -> Self::Output {
            let result  = self.gridvals.vals() + other;
            
            GridScalar {
                spec: Rc::clone(&self.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, S> std::ops::Add<&'a GridScalar<S>> for f64 {
        type Output = GridScalar<S>;

        fn add(self, other: &'a GridScalar<S>) -> Self::Output {
            let result = self + other.gridvals.vals();
            
            GridScalar {
                spec: Rc::clone(&other.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, 'b, S> std::ops::Mul<&'b GridScalar<S>> for &'a GridScalar<S> where S: GridSpec {
        type Output = GridScalar<S>;

        fn mul(self, other: &'b GridScalar<S>) -> Self::Output {
            if self.gridvals.len() == other.gridvals.len() &&
            Rc::ptr_eq(&self.get_spec(), &other.get_spec()) {
                let result = self.gridvals.vals() * other.gridvals.vals();
                GridScalar {
                    spec: Rc::clone(&self.spec),
                    gridvals: ValVector(result),
                }
            }
            else { panic!("Error multiplying GridScalars! Ensure sizes and GridSpecs are the same.") }
        }
    }

    impl<'a, S> std::ops::Mul<f64> for &'a GridScalar<S> {
        type Output = GridScalar<S>;

        fn mul(self, other: f64) -> Self::Output {
            let result = self.gridvals.vals() * other;
            
            GridScalar {
                spec: Rc::clone(&self.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, S> std::ops::Mul<&'a GridScalar<S>> for f64 {
        type Output = GridScalar<S>;

        fn mul(self, other: &'a GridScalar<S>) -> Self::Output {
            let result = self * other.gridvals.vals();
            
            GridScalar {
                spec: Rc::clone(&other.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, 'b, S> std::ops::Div<&'b GridScalar<S>> for &'a GridScalar<S> where S: GridSpec {
        type Output = GridScalar<S>;

        fn div(self, other: &'b GridScalar<S>) -> Self::Output {
            if self.gridvals.len() == other.gridvals.len() &&
            Rc::ptr_eq(&self.get_spec(), &other.get_spec()) {
                let result = self.gridvals.vals() / other.gridvals.vals();
                GridScalar {
                    spec: Rc::clone(&self.spec),
                    gridvals: ValVector(result),
                }
            }
            else { panic!("Error dividing GridScalars! Ensure sizes and GridSpecs are the same.") }
        }
    }

    impl<'a, S> std::ops::Div<f64> for &'a GridScalar<S> {
        type Output = GridScalar<S>;

        fn div(self, other: f64) -> Self::Output {
            let result = self.gridvals.vals() / other;
            
            GridScalar {
                spec: Rc::clone(&self.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, S> std::ops::Div<&'a GridScalar<S>> for f64 {
        type Output = GridScalar<S>;

        fn div(self, other: &'a GridScalar<S>) -> Self::Output {
            let result = self / other.gridvals.vals();
            
            GridScalar {
                spec: Rc::clone(&other.spec),
                gridvals: ValVector(result),
            }
        }
    }

    impl<'a, S> std::ops::Neg for &'a GridScalar<S> {
        type Output = GridScalar<S>;

        fn neg(self) -> Self::Output {
            GridScalar{
                spec: Rc::clone(&self.spec),
                gridvals: ValVector(-self.gridvals.vals()),
            }
        }
    }
}

pub mod stencil {
    
    extern crate ndarray;
    extern crate ndarray_linalg;
    extern crate factorial;

    /// The FdWeights struct contains the Stencil of points to use for a
    /// finite difference approximation, the order of the derivative to
    /// be approximated, and the "weights," or coefficients, that will
    /// be multiplied by the values at the stencil points.
    #[derive(Clone, Debug, PartialEq)]
    pub struct FdWeights {
        /// The Stencil contains the relative positions of points to be
        /// used in the finite difference approximation
        stencil: Stencil,
        /// The order of the derivative to be approximated (1 -> d, 
        /// 2 -> d2, etc.)
        nderiv: usize,
        /// The accuracy of the approximation = number of stencil points
        /// minus the derivative order
        accuracy: usize,
        /// The finite difference coefficients, or weights, contained in
        /// a ValVector
        weights: crate::grid::ValVector,
    }

    impl FdWeights {
        /// Returns an FdWeights instance fully formed and populated with
        /// the calculated coefficients. First creates a new Stencil
        /// instance with Stencil::new(), which sorts and removes duplicate
        /// values from the `slots` argument.
        /// # Arguments
        /// * `slots` - an array of integers representing the relative positions
        /// of the neighboring points to be used in the approximation; this
        /// argument will be sorted purged of duplicate values
        /// * `nderiv` - the order of the derivative to be approximated
        /// # Examples
        /// ```
        /// use rustencils::stencil::FdWeights;
        /// let s = [-2,-1,0,1,2];
        /// let d1 = FdWeights::new(&s[..], 1);
        /// let d3 = FdWeights::new(&s[..], 3);
        /// assert_eq!(d1.get_slots(), d3.get_slots());
        /// ```
        /// 
        /// ```should_panic
        /// use rustencils::stencil::FdWeights;
        /// let s = [-1,0,1];
        /// // panics because nderiv is not less than the length of s
        /// let d3 = FdWeights::new(&s[..], 3);
        /// ```
        /// 
        /// ```
        /// use rustencils::stencil::FdWeights;
        /// let s = [3,1,0,1,-1,-2,3];
        /// let d2 = FdWeights::new(&s[..], 2);
        /// assert_ne!(d2.get_slots(), &s[..]);
        /// assert_eq!(d2.get_slots(), &[-2,-1,0,1,3]);
        /// ```
        /// 
        /// ```
        /// use rustencils::stencil::FdWeights;
        /// let s = [-1,0,1];
        /// let d1 = FdWeights::new(&s[..], 1);
        /// let d2 = FdWeights::new(&s[..], 2);
        /// assert_eq!(d1.get_weights()[0], -0.5);
        /// assert_eq!(d2.get_weights()[0], 1.);
        /// ```
        pub fn new(slots: &[isize], nderiv: usize) -> Self {
            let stncl = Stencil::new(slots);
            FdWeights {
                weights: crate::grid::ValVector(Self::gen_fd_weights(&stncl, nderiv)),
                accuracy: stncl.num_slots - nderiv,
                nderiv,
                stencil: stncl,
            }
        }

        /// Solves a basic linear algebra problem to find the finite
        /// difference coefficients for arbitrary stencil points. See:
        /// https://en.wikipedia.org/wiki/Finite_difference_coefficient
        fn gen_fd_weights(stencil: &Stencil, nderiv: usize) -> ndarray::Array1<f64> {
            assert!(nderiv < stencil.num_slots,
                    "Derivative order must be less than number of stencil points!");
            let matx = Self::init_matrix(&stencil.slot_pos[..]);
            let mut bvec = ndarray::Array1::<f64>::zeros(stencil.num_slots);
            bvec[[nderiv]] = factorial::Factorial::factorial(&nderiv) as f64;
            ndarray_linalg::Solve::solve_into(&matx, bvec).unwrap()
        }

        /// Constructs the square matrix for use in generating the finite
        /// difference coefficients. Each row of the matrix is the set of
        /// stencil points raised to the power of the row index.
        fn init_matrix(slots: &[isize]) -> ndarray::Array2<f64> {
            let mut result = ndarray::Array2::<f64>::zeros((slots.len(), slots.len()));
            for i in 0..slots.len() {
                for (j, elm) in slots.iter().enumerate() {
                    result[[i,j]] = elm.pow(i as u32) as f64;
                }
            }
            result
        }

        /// Returns a shared array slice containing the stencil point
        /// positions.
        pub fn get_slots(&self) -> &[isize] {
            &self.stencil.get_slots()
        }

        /// Returns the value of the derivative order
        pub fn get_ord(&self) -> usize {
            self.nderiv
        }

        /// Returns a shared reference to a rustencils::grid::ValVector
        /// that contains the calculated finite difference coefficients
        pub fn get_weights(&self) -> &crate::grid::ValVector {
            &self.weights
        }
    }

    /// The Stencil struct represents the stencil of points that will be used
    /// to approximate some derivative. It simply contains a vector of the
    /// stencil slot positions and the number of slots.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Stencil {
        /// A vector of the relative positions of the points to be used
        slot_pos: Vec<isize>,
        /// The length of the `slot_pos` vector
        num_slots: usize,
    }

    impl Stencil {
        /// Returns a new Stencil instance. First sorts and removes duplicate
        /// values from the input argument.
        /// # Arguments
        /// * `slots` - an array of integers representing the relative positions
        /// of the neighboring points to be used in the approximation; this
        /// argument will be sorted purged of duplicate values
        fn new(slots: &[isize]) -> Self {
            let mut slots_vec = Vec::from(slots);
            slots_vec.sort_unstable();
            slots_vec.dedup();
            Stencil {
                num_slots: slots.len(),
                slot_pos: slots_vec,
            }
        }

        /// Returns a shared array slice containing the stencil point
        /// positions.
        fn get_slots(&self) -> &[isize] {
            &self.slot_pos[..]
        }
    }
    
    #[test]
    fn check_slots() {
        let slots = [3,1,0,1,-1,-2,3];
        let stncl = Stencil::new(&slots[..]);
        assert_ne!(stncl.get_slots(), &slots[..]);
        assert_eq!(stncl.get_slots(), &[-2,-1,0,1,3]);
    }
}

pub mod operator {
    
    extern crate ndarray;
    
    /// The full 1D operator construction with finite difference weights
    /// corresponding to the interior region, as well as those for the edges.
    /// The basis direction refers to the dimension along which the
    /// operator should be applied (e.g., [0][1][2] corresponding to
    /// [x][y][z] in Cartesian coordinates
    #[derive(Debug)]
    pub struct Operator1D<E> {
        /// An instance of rustencils::stencil::FdWeights holding the finite
        /// difference coefficients used on the interior region of the grid
        interior: crate::stencil::FdWeights,
        /// A set of rustencils::stencil::FdWeights instances holding the
        /// finite difference coefficients used at the edges of the grid
        /// where the interior stencil will not fit
        edge: E,
        /// Axis with respect to which the derivative will be taken (e.g.,
        /// 0 -> d/dx, 1 -> d/dy, etc.)
        basis_direction: usize,
        /// The order of the derivative
        deriv_ord: usize,
    }

    impl<E> Operator1D<E> where E: EdgeOperator {
        /// Returns a new Operator1D instance. Ensures that edge and interior
        /// FdWeights instances are all calculated for the same derivative
        /// order. Also calls the EdgeOperator `check_edges()` function to
        /// verify proper edge construction.
        /// # Arguments
        /// * `interior` - FdWeights instance that will be used for the
        /// interior region of the grid
        /// * `edge` - an object that implements EdgeOperator that holds
        /// FdWeights instances used for the grid edges
        /// * `direction` - the axis with respect to which the derivative
        /// will be taken (e.g., 0 -> d/dx, 1 -> d/dy, etc.)
        /// # Examples
        /// ```
        /// use std::rc::Rc;
        /// use rustencils::stencil::FdWeights;
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// use rustencils::operator::{Operator1D, FixedEdgeOperator};
        /// 
        /// // First initialize the grid objects
        /// let x_init = AxisSetup::new(0., 0.01, 100);
        /// let y_init = x_init.clone();
        /// let axs_init = vec![x_init, y_init];
        /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
        /// let temperature = GridScalar::zeros(Rc::clone(&spec));
        /// 
        /// // Next construct the interior and edge arguments for a 2nd order derivative
        /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
        /// 
        /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
        /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
        /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
        /// 
        /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
        /// let wts_2nd_R1 = FdWeights::new(&[-3,-2,-1,0,1], 2);
        /// let wts_2nd_R = vec![wts_2nd_R0, wts_2nd_R1];
        /// 
        /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
        /// 
        /// // Next construct the full Operator1D instances
        /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 0);
        /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 1);
        /// ```
        /// 
        /// ```should_panic
        /// use rustencils::stencil::FdWeights;
        /// use rustencils::operator::{Operator1D, FixedEdgeOperator};
        /// 
        /// // Construct the interior and edge arguments for a 2nd order derivative
        /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
        /// 
        /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
        /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
        /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
        /// 
        /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
        /// let wts_2nd_R = vec![wts_2nd_R0];
        /// 
        /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
        /// 
        /// // Next construct the full Operator1D instances
        /// // Panics because the right edge does not have enough FdWeights!
        /// // Remember that each FdWeights in the edge is only applied once
        /// // and they are applied from the outside of the grid to the interior
        /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 0);
        /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 1);
        /// ```
        pub fn new(interior: crate::stencil::FdWeights, edge: E, direction: usize) -> Self {
            let deriv_ord = interior.get_ord();
            for elm in edge.get_left() {
                assert_eq!(deriv_ord, elm.get_ord());
            }
            for elm in edge.get_right() {
                assert_eq!(deriv_ord, elm.get_ord());
            }
            let _ = edge.check_edges(&interior);
            Operator1D {
                interior,
                edge,
                basis_direction: direction,
                deriv_ord,
            }
        }

        /// Retruns the order of the derivative
        pub fn get_ord(&self) -> usize {
            self.deriv_ord
        }
    }

    /// Since the edges of the grid can be defined in multiple ways (e.g.,
    /// non-periodic -- called "fixed" in this crate -- vs periodic) the
    /// EdgeOperator trait is used to identify those structs that can be
    /// used for this purpose. The trait defines the necessary methods for
    /// a struct that specifies a type of grid edge construction. For
    /// example, this trait is implemented by the FixedEdgeOperator type.
    /// NOTE: that the boundary conditions are separate from the edge
    /// operator and are specified elsewhere.
    pub trait EdgeOperator {
        /// Returns a Result<(), &'static str> depending on whether the edges
        /// are constructed properly. It is also valid to simply panic when
        /// the edge construction is not correct. Generally advisable to just
        /// include logic about when to check which edge and conditionally
        /// call `check_left_edge` and `check_right_edge`.
        /// # Arguments
        /// * `weights_int` - the corresponding FdWeights instance used for
        /// the interior of the grid
        fn check_edges(&self, weights_int: &crate::stencil::FdWeights) -> Result<(), &'static str>;
        /// Check and assert the construction of the left edge is correct.
        /// Panics if it is not correct.
        /// # Arguments
        /// * `weights_int` - the corresponding FdWeights instance used for
        /// the interior of the grid
        fn check_left_edge(&self, weights_int: &crate::stencil::FdWeights);
        /// Check and assert the construction of the right edge is correct.
        /// Panics if it is not correct.
        /// # Arguments
        /// * `weights_int` - the corresponding FdWeights instance used for
        /// the interior of the grid
        fn check_right_edge(&self, weights_int: &crate::stencil::FdWeights);
        /// Returns an exclusive reference to the vector of left edge
        /// FdWeights
        fn get_left_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights>;
        /// Returns an exclusive reference to the vector of right edge
        /// FdWeights
        fn get_right_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights>;
        /// Returns a shared reference to the vector of left edge FdWeights
        fn get_left(&self) -> &Vec<crate::stencil::FdWeights>;
        /// Returns a shared reference to the vector of right edge FdWeights
        fn get_right(&self) -> &Vec<crate::stencil::FdWeights>;
    }

    /// The FixedEdgeOperator struct contains vectors of rustencils::stencil::FdWeights.
    /// One vector for the left edge and one vector for the right edge.
    /// NOTE: "Fixed" refers to the fact that the bounds are NOT periodic! The
    /// boundary conditions can still be of any type and must be specified separately!
    /// The left (more negative side) and right (more positive side) edge operators will
    /// be applied from the outside-in (i.e. the first element in the vector will apply
    /// to the outermost point, and so on) and each element is only applied once. The
    /// user is responsible for ensuring adequate edge operator construction given the
    /// structure of the interior operator.
    #[derive(Clone, Debug)]
    pub struct FixedEdgeOperator {
        /// The type of edge operator. Constructor sets this to "fixed"
        edge_type: String,
        /// A vector of FdWeights to be applied to the left edge
        left: Vec<crate::stencil::FdWeights>,
        /// A vector of FdWeights to be applied to the right edge
        right: Vec<crate::stencil::FdWeights>,
    }

    impl EdgeOperator for FixedEdgeOperator {
        fn check_edges(&self, weights_int: &crate::stencil::FdWeights) -> Result<(), &'static str> {
            match weights_int.get_slots().iter().min() {
                Some(x) if x < &0 => self.check_left_edge(weights_int),
                Some(_) => {},
                None => {}
            }
            match weights_int.get_slots().iter().max() {
                Some(x) if x > &0 => self.check_right_edge(weights_int),
                Some(_) => {},
                None => {}
            }
            Ok(())
        }

        fn check_left_edge(&self, weights_int: &crate::stencil::FdWeights) {
            assert_eq!(weights_int.get_slots().iter().min().unwrap(), &-(self.left.len() as isize), "Improper number of left edge stencils!");
            for (n, item) in self.left.iter().enumerate() {
                assert!(item.get_slots().iter().min().unwrap() >= &(0-(n as isize)), "Edge stencil out of range!");
            }
        }

        fn check_right_edge(&self, weights_int: &crate::stencil::FdWeights) {
            assert_eq!(weights_int.get_slots().iter().max().unwrap(), &(self.right.len() as isize), "Improper number of right edge stencils!");
            for (n, item) in self.right.iter().enumerate() {
                assert!(item.get_slots().iter().max().unwrap() <= &(n as isize), "Edge stencil out of range!");
            }
        }

        fn get_left_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights> { &mut self.left }
        fn get_right_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights> { &mut self.right }
        fn get_left(&self) -> &Vec<crate::stencil::FdWeights> { &self.left }
        fn get_right(&self) -> &Vec<crate::stencil::FdWeights> { &self.right }
    }

    impl FixedEdgeOperator {
        /// Returns a FixedEdgeOperator. Sets the `edge_type` to "fixed".
        /// # Arguments
        /// * `left_edge_ops` - a vector of FdWeights for the left edge
        /// * `right_edge_ops` - a vector of FdWeights for the right edge
        /// # Examples
        /// ```
        /// use rustencils::stencil::FdWeights;
        /// use rustencils::operator::{EdgeOperator, FixedEdgeOperator};
        /// 
        /// // Construct the interior and edge arguments for a 2nd order derivative
        /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
        /// 
        /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
        /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
        /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
        /// 
        /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
        /// let wts_2nd_R1 = FdWeights::new(&[-3,-2,-1,0,1], 2);
        /// let wts_2nd_R = vec![wts_2nd_R0, wts_2nd_R1];
        /// 
        /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
        /// 
        /// edge_wts_2nd.check_edges(&wts_2nd_int);
        /// ```
        pub fn new(left_edge_ops: Vec<crate::stencil::FdWeights>,
                   right_edge_ops: Vec<crate::stencil::FdWeights>) -> Self {
            FixedEdgeOperator {
                edge_type: String::from("fixed"),
                left: left_edge_ops,
                right: right_edge_ops,
            }
        }
    }

    /// The OperatorMatrix struct represents the 2D matrix linear operator
    /// approximating some derivative. Holds the matrix and the shape of
    /// the matrix.
    pub struct OperatorMatrix {
        /// The shape of the matrix: (rows, columns)
        shape: (usize, usize),
        /// The actual matrix, here implemented with ndarray
        matrix: ndarray::Array2<f64>,
    }

    impl OperatorMatrix {
        /// Returns a new GridQty that is the result of taking the inner
        /// product of the OperatorMatrix with the GridQty passed in as
        /// argument. This is the approximation of taking a derivative.
        /// # Arguments
        /// * `qty` - an object to be differentiated that implements
        /// rustencils::grid::GridQty
        /// # Examples
        /// ```
        /// use std::rc::Rc;
        /// use rustencils::stencil::FdWeights;
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
        /// use rustencils::operator::construct_op;
        /// 
        /// // First initialize the grid objects
        /// let x_init = AxisSetup::new(0., 0.01, 100);
        /// let y_init = x_init.clone();
        /// let axs_init = vec![x_init, y_init];
        /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
        /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 0);
        /// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 1);
        /// // T will represent temperature
        /// let T = 100. * &( &( &x_vals * &x_vals) * &( &y_vals * &y_vals) );
        /// 
        /// // Next construct the interior and edge arguments for a 2nd order derivative
        /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
        /// 
        /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
        /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
        /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
        /// 
        /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
        /// let wts_2nd_R1 = FdWeights::new(&[-3,-2,-1,0,1], 2);
        /// let wts_2nd_R = vec![wts_2nd_R0, wts_2nd_R1];
        /// 
        /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
        /// 
        /// // Next construct the full Operator1D instances
        /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 0);
        /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 1);
        /// 
        /// // Construct OperatorMatrix instances
        /// let d2dx2 = construct_op(op1d_2nd_x, &T);
        /// let d2dy2 = construct_op(op1d_2nd_y, &T);
        /// 
        /// // Differentiate T
        /// let d2Tdx2 = d2dx2.of_qty(&T);
        /// let d2Tdy2 = d2dy2.of_qty(&T);
        /// 
        /// // Can also construct more complex operator!
        /// let Del2 = &d2dx2 + &d2dy2;
        /// let Del2T = Del2.of_qty(&T);
        /// ```
        pub fn of_qty<Q, S>(&self, qty: &Q) -> Q
        where Q: GridQty<S>, S: GridSpec {
            assert_eq!(qty.get_gridvals().len(), self.shape.0);
            let result = self.matrix.dot(qty.get_gridvals().as_ndarray());
            GridQty::new(qty.get_spec(), crate::grid::ValVector(result))
        }

        /// Returns a new OperatorMatrix that is the result of taking the
        /// inner product of the OperatorMatrix (self) with the
        /// OperatorMarix passed in as argument.
        /// # Arguments
        /// * `other` - another OperatorMatrix instance
        /// # Examples
        /// ```
        /// use std::rc::Rc;
        /// use rustencils::stencil::FdWeights;
        /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
        /// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
        /// use rustencils::operator::construct_op;
        /// 
        /// // First initialize the grid objects
        /// let x_init = AxisSetup::new(0., 0.01, 100);
        /// let y_init = x_init.clone();
        /// let axs_init = vec![x_init, y_init];
        /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
        /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 0);
        /// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 1);
        /// // T will represent temperature
        /// let T = 100. * &( &( &x_vals * &x_vals) * &( &y_vals * &y_vals) );
        /// 
        /// // Next construct the interior and edge arguments for a 2nd order derivative
        /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
        /// 
        /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
        /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
        /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
        /// 
        /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
        /// let wts_2nd_R1 = FdWeights::new(&[-3,-2,-1,0,1], 2);
        /// let wts_2nd_R = vec![wts_2nd_R0, wts_2nd_R1];
        /// 
        /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
        /// 
        /// // Next construct the full Operator1D instances
        /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 0);
        /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 1);
        /// 
        /// // Construct OperatorMatrix instances
        /// let d2dx2 = construct_op(op1d_2nd_x, &T);
        /// let d2dy2 = construct_op(op1d_2nd_y, &T);
        /// 
        /// // Differentiate T
        /// let d2Tdy2 = d2dy2.of_qty(&T);
        /// let d4Tdx2dy2 = d2dx2.of_qty(&d2Tdy2);
        /// 
        /// // Can also construct more complex operator!
        /// let d4dx2dy2 = d2dx2.of_mtx(&d2dy2);
        /// let d4Tdx2dy2 = d4dx2dy2.of_qty(&T);
        /// ```
        pub fn of_mtx(&self, other: &OperatorMatrix) -> Self {
            if self.shape == other.shape {
                let result = self.matrix.dot(&other.matrix);
                OperatorMatrix {
                    shape: self.shape,
                    matrix: result,
                }
            }
            else{ panic!("Error taking inner product of OperatorMatrix! Ensure shapes are the same.") }
        }
    }

    use crate::grid::{GridQty, GridSpec};

    /// Constructs a new OperatorMatrix based on an Operator1D and a GridQty
    /// # Arguments
    /// * `op1d` - an Operator1D instance
    /// * `qty` - an instance of something that implements GridQty
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use rustencils::stencil::FdWeights;
    /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
    /// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
    /// use rustencils::operator::construct_op;
    /// 
    /// // First initialize the grid objects
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let axs_init = vec![x_init, y_init];
    /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
    /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 0);
    /// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 1);
    /// // T will represent temperature
    /// let T = 100. * &( &( &x_vals * &x_vals) * &( &y_vals * &y_vals) );
    /// 
    /// // Next construct the interior and edge arguments for a 2nd order derivative
    /// let wts_2nd_int = FdWeights::new(&[-2,-1,0,1,2], 2);
    /// 
    /// let wts_2nd_L0 = FdWeights::new(&[0,1,2,3,4], 2);
    /// let wts_2nd_L1 = FdWeights::new(&[-1,0,1,2,3], 2);
    /// let wts_2nd_L = vec![wts_2nd_L0, wts_2nd_L1];
    /// 
    /// let wts_2nd_R0 = FdWeights::new(&[-4,-3,-2,-1,0], 2);
    /// let wts_2nd_R1 = FdWeights::new(&[-3,-2,-1,0,1], 2);
    /// let wts_2nd_R = vec![wts_2nd_R0, wts_2nd_R1];
    /// 
    /// let edge_wts_2nd = FixedEdgeOperator::new(wts_2nd_L, wts_2nd_R);
    /// 
    /// // Next construct the full Operator1D instances
    /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 0);
    /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 1);
    /// 
    /// // Construct OperatorMatrix instances
    /// let d2dx2 = construct_op(op1d_2nd_x, &T);
    /// let d2dy2 = construct_op(op1d_2nd_y, &T);
    /// ```
    pub fn construct_op<Q, S, E>(op1d: Operator1D<E>, qty: &Q) -> OperatorMatrix
    where Q: GridQty<S>, S: GridSpec, E: EdgeOperator
    {
        let dim_num = op1d.basis_direction;
        let dim_pts = qty.get_spec().get_gridshape()[dim_num];
        let tot_pts = qty.get_gridvals().len();
        let shape = (tot_pts, tot_pts);
        let deriv_ord = op1d.get_ord();
        let denom = (qty.get_spec().get_spacing()[dim_num]).powi(deriv_ord as i32);
        let mut matrix: ndarray::Array2<f64> = ndarray::Array2::zeros(shape);
        for (idxs, pt) in qty.get_grid().indexed_iter() {
            let left_idx = idxs[dim_num];
            let right_idx = dim_pts - idxs[dim_num] - 1;
            
            let (stncl, weights) = match (left_idx, right_idx) {
                (left_idx, right_idx) if left_idx >= op1d.edge.get_left().len() && right_idx >= op1d.edge.get_right().len()
                            => (op1d.interior.get_slots(), op1d.interior.get_weights()),
                (left_idx, _) if left_idx < op1d.edge.get_left().len()
                            => (op1d.edge.get_left()[left_idx].get_slots(), op1d.edge.get_left()[left_idx].get_weights()),
                (_, right_idx) if right_idx < op1d.edge.get_right().len()
                            => (op1d.edge.get_right()[right_idx].get_slots(), op1d.edge.get_right()[right_idx].get_weights()),
                (_, _) => panic!("Error while constructing operator!"),
            };

            let _ = stncl.iter().enumerate().map(|(i, rel_pos)| {
                let mut new_idxs = idxs.clone();
                new_idxs[dim_num] = (new_idxs[dim_num] as isize + rel_pos) as usize;
                let mtx_col_idx = qty.get_grid()[new_idxs].idx;
                matrix[[pt.idx, mtx_col_idx]] = weights[i]/denom;
            }).collect::<()>();
        }
        
        OperatorMatrix {
            shape,
            matrix,
        }
    }

    impl<'a, 'b> std::ops::Add<&'b OperatorMatrix> for &'a OperatorMatrix {
        type Output = OperatorMatrix;

        fn add(self, other: &'b OperatorMatrix) -> Self::Output {
            if self.shape == other.shape {
                let result = &self.matrix + &other.matrix;
                OperatorMatrix {
                    shape: self.shape,
                    matrix: result,
                }
            }
            else { panic!("Error adding OperatorMatrix instances! Ensure shapes are the same.") }
        }
    }

    impl<'a, 'b> std::ops::Sub<&'b OperatorMatrix> for &'a OperatorMatrix {
        type Output = OperatorMatrix;

        fn sub(self, other: &'b OperatorMatrix) -> Self::Output {
            if self.shape == other.shape {
                let result = &self.matrix - &other.matrix;
                OperatorMatrix {
                    shape: self.shape,
                    matrix: result,
                }
            }
            else { panic!("Error subtracting OperatorMatrix instances! Ensure shapes are the same.") }
        }
    }
}

pub mod boundaries {
    pub trait BoundaryHandler {
        fn set_bounds(); // maybe call this set_BCs?
        fn check_bc_type();
    }

    pub trait BoundaryCondition {
        fn print_bc(); // print the conatained BC
        fn get_boundary(); // return the slice of the grid corresponding to this boundary
        fn get_bc_type();
    }

    pub enum BoundarySide {
        Low,
        High,
    }

    pub struct DirichletHandler<T> {
        bc_type: String,
        bc_list: Vec<T>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
