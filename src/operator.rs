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
    /// Axis with respect to which the derivative will be taken. Must
    /// match the character used to construct the axis.
    basis_direction: char,
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
    /// use std::collections::HashMap;
    /// use rustencils::stencil::FdWeights;
    /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
    /// use rustencils::operator::{Operator1D, FixedEdgeOperator};
    /// 
    /// // First initialize the grid objects
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let mut axs_init = HashMap::new();
    /// axs_init.insert('x', x_init);
    /// axs_init.insert('y', y_init);
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
    /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 'x');
    /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 'y');
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
    /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 'x');
    /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 'y');
    /// ```
    pub fn new(interior: crate::stencil::FdWeights, edge: E, axis: char) -> Self {
        let deriv_ord = interior.ord();
        for elm in edge.left() {
            assert_eq!(deriv_ord, elm.ord());
        }
        for elm in edge.right() {
            assert_eq!(deriv_ord, elm.ord());
        }
        let _ = edge.check_edges(&interior);
        Operator1D {
            interior,
            edge,
            basis_direction: axis,
            deriv_ord,
        }
    }

    /// Retruns the order of the derivative
    pub fn ord(&self) -> usize {
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
    fn left_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights>;
    /// Returns an exclusive reference to the vector of right edge
    /// FdWeights
    fn right_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights>;
    /// Returns a shared reference to the vector of left edge FdWeights
    fn left(&self) -> &Vec<crate::stencil::FdWeights>;
    /// Returns a shared reference to the vector of right edge FdWeights
    fn right(&self) -> &Vec<crate::stencil::FdWeights>;
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
        match weights_int.slots().iter().min() {
            Some(x) if x < &0 => self.check_left_edge(weights_int),
            Some(_) => {},
            None => {}
        }
        match weights_int.slots().iter().max() {
            Some(x) if x > &0 => self.check_right_edge(weights_int),
            Some(_) => {},
            None => {}
        }
        Ok(())
    }

    fn check_left_edge(&self, weights_int: &crate::stencil::FdWeights) {
        assert_eq!(weights_int.slots().iter().min().unwrap(), &-(self.left.len() as isize), "Improper number of left edge stencils!");
        for (n, item) in self.left.iter().enumerate() {
            assert!(item.slots().iter().min().unwrap() >= &(0-(n as isize)), "Edge stencil out of range!");
        }
    }

    fn check_right_edge(&self, weights_int: &crate::stencil::FdWeights) {
        assert_eq!(weights_int.slots().iter().max().unwrap(), &(self.right.len() as isize), "Improper number of right edge stencils!");
        for (n, item) in self.right.iter().enumerate() {
            assert!(item.slots().iter().max().unwrap() <= &(n as isize), "Edge stencil out of range!");
        }
    }

    fn left_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights> { &mut self.left }
    fn right_mut(&mut self) -> &mut Vec<crate::stencil::FdWeights> { &mut self.right }
    fn left(&self) -> &Vec<crate::stencil::FdWeights> { &self.left }
    fn right(&self) -> &Vec<crate::stencil::FdWeights> { &self.right }
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
    /// use std::collections::HashMap;
    /// use rustencils::stencil::FdWeights;
    /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
    /// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
    /// use rustencils::operator::construct_op;
    /// 
    /// // First initialize the grid objects
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let mut axs_init = HashMap::new();
    /// axs_init.insert('x', x_init);
    /// axs_init.insert('y', y_init);
    /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
    /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 'x');
    /// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 'y');
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
    /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 'x');
    /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 'y');
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
        assert_eq!(qty.gridvals().len(), self.shape.0);
        let result = self.matrix.dot(qty.gridvals().as_ndarray());
        GridQty::new(qty.spec(), crate::grid::ValVector(result))
    }

    /// Returns a new OperatorMatrix that is the result of taking the
    /// inner product of the OperatorMatrix (self) with the
    /// OperatorMarix passed in as argument.
    /// # Arguments
    /// * `other` - another OperatorMatrix instance
    /// # Examples
    /// ```
    /// use std::rc::Rc;
    /// use std::collections::HashMap;
    /// use rustencils::stencil::FdWeights;
    /// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
    /// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
    /// use rustencils::operator::construct_op;
    /// 
    /// // First initialize the grid objects
    /// let x_init = AxisSetup::new(0., 0.01, 100);
    /// let y_init = x_init.clone();
    /// let mut axs_init = HashMap::new();
    /// axs_init.insert('x', x_init);
    /// axs_init.insert('y', y_init);
    /// let spec = Rc::new(CartesianGridSpec::new(axs_init));
    /// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 'x');
    /// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 'y');
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
    /// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 'x');
    /// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 'y');
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
/// use std::collections::HashMap;
/// use rustencils::stencil::FdWeights;
/// use rustencils::grid::{GridScalar, GridQty, CartesianGridSpec, AxisSetup};
/// use rustencils::operator::{Operator1D, FixedEdgeOperator, OperatorMatrix};
/// use rustencils::operator::construct_op;
/// 
/// // First initialize the grid objects
/// let x_init = AxisSetup::new(0., 0.01, 100);
/// let y_init = x_init.clone();
/// let mut axs_init = HashMap::new();
/// axs_init.insert('x', x_init);
/// axs_init.insert('y', y_init);
/// let spec = Rc::new(CartesianGridSpec::new(axs_init));
/// let x_vals = GridScalar::axis_vals(Rc::clone(&spec), 'x');
/// let y_vals = GridScalar::axis_vals(Rc::clone(&spec), 'y');
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
/// let op1d_2nd_x = Operator1D::new(wts_2nd_int.clone(), edge_wts_2nd.clone(), 'x');
/// let op1d_2nd_y = Operator1D::new(wts_2nd_int, edge_wts_2nd, 'y');
/// 
/// // Construct OperatorMatrix instances
/// let d2dx2 = construct_op(op1d_2nd_x, &T);
/// let d2dy2 = construct_op(op1d_2nd_y, &T);
/// ```
pub fn construct_op<Q, S, E>(op1d: Operator1D<E>, qty: &Q) -> OperatorMatrix
where Q: GridQty<S>, S: GridSpec, E: EdgeOperator
{
    let dim = op1d.basis_direction;
    let dim_pts = qty.spec().gridshape()[&dim];
    let tot_pts = qty.gridvals().len();
    let shape = (tot_pts, tot_pts);
    let deriv_ord = op1d.ord();
    let denom = (qty.spec().spacing()[&dim]).powi(deriv_ord as i32);
    let mut matrix: ndarray::Array2<f64> = ndarray::Array2::zeros(shape);
    for (idxs, pt) in qty.grid().indexed_iter() {
        let left_idx = idxs[qty.spec().grid_axes()[&dim]];
        let right_idx = dim_pts - idxs[qty.spec().grid_axes()[&dim]] - 1;
        
        let (stncl, weights) = match (left_idx, right_idx) {
            (left_idx, right_idx) if left_idx >= op1d.edge.left().len() && right_idx >= op1d.edge.right().len()
                        => (op1d.interior.slots(), op1d.interior.weights()),
            (left_idx, _) if left_idx < op1d.edge.left().len()
                        => (op1d.edge.left()[left_idx].slots(), op1d.edge.left()[left_idx].weights()),
            (_, right_idx) if right_idx < op1d.edge.right().len()
                        => (op1d.edge.right()[right_idx].slots(), op1d.edge.right()[right_idx].weights()),
            (_, _) => panic!("Error while constructing operator!"),
        };

        let _ = stncl.iter().enumerate().map(|(i, rel_pos)| {
            let mut new_idxs = idxs.clone();
            new_idxs[qty.spec().grid_axes()[&dim]] = (new_idxs[qty.spec().grid_axes()[&dim]] as isize + rel_pos) as usize;
            let mtx_col_idx = qty.grid()[new_idxs].idx;
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