pub mod grid {

    extern crate ndarray;

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
    
    pub trait GridSpec {
        fn get_coords(&self) -> &Vec<Vec<f64>>;
        fn get_ndim(&self) -> usize;
        fn get_gridshape(&self) -> &Vec<usize>;
        fn get_spacing(&self) -> &Vec<f64>;
        fn get_grid(&self) -> &Grid;
        // fn clone(&self) -> Self;
    }

    pub struct CartesianGridSpec {
        coords: Vec<Vec<f64>>, // e.g. [[x0,x1,x2,x3,x4,...,xm],[y0,y1,y2,y3,y4,...,yn]]
        ndim: usize,           // e.g. 2
        gridshape: Vec<usize>, // e.g. [m,n]
        spacing: Vec<f64>,     // e.g. [(x1-x0),(y1-y0)]
        grid: Grid,
    }

    impl GridSpec for CartesianGridSpec {
        fn get_coords(&self) -> &Vec<Vec<f64>> { &self.coords }
        fn get_ndim(&self) -> usize { self.ndim }
        fn get_gridshape(&self) -> &Vec<usize> { &self.gridshape }
        fn get_spacing(&self) -> &Vec<f64> { &self.spacing }
        fn get_grid(&self) -> &Grid { &self.grid }
    }

    #[derive(Clone)]
    pub struct AxisSetup {
        start: f64,
        delta: f64,
        steps: usize,
    }

    impl AxisSetup {
        pub fn new(start: f64, delta: f64, steps: usize) -> Self {
            AxisSetup {
                delta: delta.abs(),
                start,
                steps,
            }
        }
    }

    #[derive(Default)]
    pub struct Point {
        pub coord: Vec<f64>,
        pub idx: usize,
    }

    impl CartesianGridSpec {
        // Potential issue with casting usize to f32 if high precision required for
        // axis values.
        pub fn new(axes: Vec<AxisSetup>) -> Self {
            let gridshape: Vec<usize> = axes.iter().map(|ax| ax.steps).collect();
            let coords: Vec<Vec<f64>> = axes.iter().map(|ax| {
                    let mut set = Vec::with_capacity(ax.steps);
                    for i in 0..ax.steps {set.push(ax.start + (i as f64)*ax.delta);}
                    set
                }).collect();
            let spacing: Vec<f64> = axes.iter().map(|ax| ax.delta).collect();
            let mut grid: ndarray::ArrayD<Point> = ndarray::Array::default(gridshape.clone());
            let mut count = 0;
            let _ = grid.indexed_iter_mut().map(|(indices,pt)| {
                pt.coord = Vec::new();
                for i in 0..coords.len() {
                    pt.coord.push(coords[i][indices[i]]);
                }
                pt.idx = count;
                count += 1;
            }).collect::<()>();
            CartesianGridSpec {
                spacing: spacing,
                gridshape: gridshape,
                ndim: axes.len(),
                coords: coords,
                grid: Grid(grid),
            }
        }
    }

    pub struct ValVector(pub(crate) ndarray::Array1<f64>);

    impl ValVector {
        pub fn len(&self) -> usize {
            self.0.len()
        }

        fn vals(&self) -> &ndarray::Array1<f64> {
            &self.0
        }

        pub(crate) fn into_ndarray(&self) -> &ndarray::Array1<f64> {
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
    
    pub trait GridQty<T> {
        fn get_spec(&self) -> Rc<T>;
        fn get_gridvals(&self) -> &ValVector;
        fn get_grid(&self) -> &Grid;
        fn new(spec: Rc<T>, gridvals: ValVector) -> Self;
    }

    use std::rc::Rc;
    
    pub struct GridScalar<T> {
        spec: Rc<T>,
        gridvals: ValVector, // A vector that just contains the values of interest at every point
    }

    impl<T> GridQty<T> for GridScalar<T> where T: GridSpec {
        fn get_spec(&self) -> Rc<T> { Rc::clone(&self.spec) }
        fn get_gridvals(&self) -> &ValVector { &self.gridvals }
        fn get_grid(&self) -> &Grid { self.spec.get_grid() }
        fn new(spec: Rc<T>, gridvals: ValVector) -> Self {
            Self::new(spec, gridvals)
        }
    }

    impl<T> GridScalar<T> where T: GridSpec {
        fn new(spec: Rc<T>, gridvals: ValVector) -> Self {
            GridScalar {
                spec: spec,
                gridvals: gridvals,
            }
        }

        pub fn uniform(spec: Rc<T>, value: f64) -> Self {
            let mut n = 1;
            for elm in spec.get_gridshape() {
                n *= elm;
            }
            let gridvals: ndarray::Array1<f64> = ndarray::arr1(&vec![value; n][..]);
            GridScalar{
                spec: spec,
                gridvals: ValVector(gridvals),
            }
        }

        pub fn ones(spec: Rc<T>) -> Self {
            GridScalar::uniform(spec, 1.)
        }

        pub fn zeros(spec: Rc<T>) -> Self {
            GridScalar::uniform(spec, 0.)
        }

        pub fn axis_vals(spec: Rc<T>, dimension: usize) -> Self {
            let mut axis = GridScalar::zeros(Rc::clone(&spec));
            let _ = spec.get_grid().iter().map(|point| {
                axis.gridvals[point.idx] = point.coord[dimension];
            }).collect::<()>();
            axis
        }
    }

    impl<T> std::ops::Add for GridScalar<T> where T: GridSpec {
        type Output = Self;

        fn add(self, other: GridScalar<T>) -> Self {
            if self.gridvals.len() != other.gridvals.len() { panic!("Can't add two grids of different sizes!") }
            if &*self.get_spec() as *const _ != &*other.get_spec() as *const _ { panic!("Grid Specs must be the same!") }

            let result = self.gridvals.vals() + other.gridvals.vals();

            GridScalar {
                spec: self.spec,
                gridvals: ValVector(result),
            }
        }
    }

    impl<T> std::ops::Add<f64> for GridScalar<T> {
        type Output = Self;

        fn add(self, other: f64) -> Self {
            let result  = self.gridvals.vals() + other;
            
            GridScalar {
                spec: self.spec,
                gridvals: ValVector(result),
            }
        }
    }

    impl<T> std::ops::Mul for GridScalar<T> where T: GridSpec {
        type Output = Self;

        fn mul(self, other: GridScalar<T>) -> Self {
            if self.gridvals.len() != other.gridvals.len() { panic!("Can't multiply two grids of different sizes!") }
            if &*self.get_spec() as *const _ != &*other.get_spec() as *const _ { panic!("Grid specs must be the same!") }

            let result = self.gridvals.vals() * other.gridvals.vals();
            
            GridScalar {
                spec: self.spec,
                gridvals: ValVector(result),
            }
        }
    }

    impl<T> std::ops::Mul<f64> for GridScalar<T> {
        type Output = Self;

        fn mul(self, other: f64) -> Self {
            let result = self.gridvals.vals() * other;
            
            GridScalar {
                spec: self.spec,
                gridvals: ValVector(result),
            }
        }
    }
}

pub mod stencil {
    
    extern crate ndarray;
    extern crate ndarray_linalg;
    extern crate factorial;

    pub struct FD_Weights {
        stencil: Stencil,
        nderiv: usize,
        accuracy: usize,
        weights: crate::grid::ValVector, // ndarray::Array1<f64>,
    }

    // Called Operator1D in APC524 Main_PDE_Repo
    impl FD_Weights {
        pub fn new(slots: &[isize], nderiv: usize) -> Self {
            let stncl = Stencil::new(slots);
            FD_Weights {
                weights: crate::grid::ValVector(Self::gen_fd_weights(&stncl, nderiv)),
                accuracy: stncl.num_slots - nderiv,
                nderiv: nderiv,
                stencil: stncl,
            }
        }

        fn gen_fd_weights(stencil: &Stencil, nderiv: usize) -> ndarray::Array1<f64> {
            assert!(nderiv < stencil.num_slots,
                    "Derivative order must be less than number of stencil points!");
            let matx = Self::init_matrix(&stencil.slot_pos[..]);
            let mut bvec = ndarray::Array1::<f64>::zeros(stencil.num_slots);
            bvec[[nderiv]] = factorial::Factorial::factorial(&nderiv) as f64;
            ndarray_linalg::Solve::solve_into(&matx, bvec).unwrap()
        }

        fn init_matrix(slots: &[isize]) -> ndarray::Array2<f64> {
            let mut result = ndarray::Array2::<f64>::zeros((slots.len(), slots.len()));
            for i in 0..slots.len() {
                for (j, elm) in slots.iter().enumerate() {
                    result[[i,j]] = elm.pow(i as u32) as f64;
                }
            }
            result
        }

        pub fn get_slots(&self) -> &[isize] {
            &self.stencil.slot_pos[..]
        }

        pub fn get_ord(&self) -> usize {
            self.nderiv
        }

        pub fn get_weights(&self) -> &crate::grid::ValVector {
            &self.weights
        }
    }

    pub struct Stencil {
        slot_pos: Vec<isize>,
        num_slots: usize,
    }

    impl Stencil {
        pub fn new(slots: &[isize]) -> Self {
            let mut slots_vec = Vec::from(slots);
            slots_vec.sort();
            Stencil {
                num_slots: slots.len(),
                slot_pos: slots_vec,
            }
        }
    }
}

pub mod operator {
    
    extern crate ndarray;
    
    // The full 1D operator construction with finite difference weights
    // corresponding to the interior region, as well as the edges.
    // The basis direction refers to the dimension along which the
    // operator should be applied: [0][1][2] corresponding to
    // [x][y][z] of GridScalar.grididxs
    pub struct Operator1D<T> {
        interior: super::stencil::FD_Weights,
        edge: T,
        basis_direction: usize,
        deriv_ord: usize,
    }

    impl<T> Operator1D<T> where T: EdgeOperator {
        pub fn new(interior: super::stencil::FD_Weights, edge: T, direction: usize) -> Self {
            let deriv_ord = interior.get_ord();
            for elm in edge.get_left() {
                assert_eq!(deriv_ord, elm.get_ord());
            }
            for elm in edge.get_right() {
                assert_eq!(deriv_ord, elm.get_ord());
            }
            edge.check_edges(&interior);
            Operator1D {
                interior: interior,
                edge: edge,
                basis_direction: direction,
                deriv_ord: deriv_ord,
            }
        }

        pub fn get_ord(&self) -> usize {
            self.deriv_ord
        }
    }

    pub trait EdgeOperator {
        fn check_edges(&self, weights_int: &super::stencil::FD_Weights) -> Result<(), &'static str>;
        fn check_left_edge(&self, weights_int: &super::stencil::FD_Weights);
        fn check_right_edge(&self, weights_int: &super::stencil::FD_Weights);
        fn get_left_mut(&mut self) -> &mut Vec<super::stencil::FD_Weights>;
        fn get_right_mut(&mut self) -> &mut Vec<super::stencil::FD_Weights>;
        fn get_left(&self) -> &Vec<super::stencil::FD_Weights>;
        fn get_right(&self) -> &Vec<super::stencil::FD_Weights>;
    }

    // NOTE: "Fixed" refers to the fact that the bounds are NOT periodic! The
    // boundary conditions can still be of any type and must be specified separately!
    // The left (more negative side) and right (more positive side) edge operators will
    // be applied from the outside-in (i.e. the first element in the vector will apply
    // to the outermost point, and so on). The user is responsible for ensuring adequate
    // edge operator construction given the structure of the interior operator.
    pub struct FixedEdgeOperator {
        edge_type: String,
        left: Vec<super::stencil::FD_Weights>,
        right: Vec<super::stencil::FD_Weights>,
    }

    impl EdgeOperator for FixedEdgeOperator {
        fn check_edges(&self, weights_int: &super::stencil::FD_Weights) -> Result<(), &'static str> {
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

        fn check_left_edge(&self, weights_int: &super::stencil::FD_Weights) {
            assert_eq!(weights_int.get_slots().iter().min().unwrap(), &(self.left.len() as isize), "Improper number of left edge stencils!");
            for (n, item) in self.left.iter().enumerate() {
                assert!(item.get_slots().iter().min().unwrap() >= &(0-(n as isize)), "Edge stencil out of range!");
            }
        }

        fn check_right_edge(&self, weights_int: &super::stencil::FD_Weights) {
            assert_eq!(weights_int.get_slots().iter().max().unwrap(), &(self.right.len() as isize), "Improper number of right edge stencils!");
            for (n, item) in self.right.iter().enumerate() {
                assert!(item.get_slots().iter().max().unwrap() <= &(n as isize), "Edge stencil out of range!");
            }
        }

        fn get_left_mut(&mut self) -> &mut Vec<super::stencil::FD_Weights> { &mut self.left }
        fn get_right_mut(&mut self) -> &mut Vec<super::stencil::FD_Weights> { &mut self.right }
        fn get_left(&self) -> &Vec<super::stencil::FD_Weights> { &self.left }
        fn get_right(&self) -> &Vec<super::stencil::FD_Weights> { &self.right }
    }

    impl FixedEdgeOperator {
        pub fn new(left_edge_ops: Vec<super::stencil::FD_Weights>,
                   right_edge_ops: Vec<super::stencil::FD_Weights>) -> Self {
            FixedEdgeOperator {
                edge_type: String::from("fixed"),
                left: left_edge_ops,
                right: right_edge_ops,
            }
        }
    }

    pub struct OperatorMatrix {
        shape: (usize, usize),
        matrix: ndarray::Array2<f64>,
        // populator: Option<MatrixPopulator>,
    }

    impl OperatorMatrix {
        pub fn of<T, U>(&self, qty: T) -> T
        where T: GridQty<U>, U: GridSpec {
            assert_eq!(qty.get_gridvals().len(), self.shape.0);
            let result = self.matrix.dot(qty.get_gridvals().into_ndarray());
            GridQty::new(qty.get_spec(), crate::grid::ValVector(result))
        }
    }

    use crate::grid::{GridQty, GridSpec};

    pub fn construct_op<T, U, V>(op1d: Operator1D<T>, qty: &U) -> OperatorMatrix
    where U: GridQty<V>, V: GridSpec, T: EdgeOperator
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
