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
    /// assert_eq!(d1.slots(), d3.slots());
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
    /// assert_ne!(d2.slots(), &s[..]);
    /// assert_eq!(d2.slots(), &[-2,-1,0,1,3]);
    /// ```
    /// 
    /// ```
    /// use rustencils::stencil::FdWeights;
    /// let s = [-1,0,1];
    /// let d1 = FdWeights::new(&s[..], 1);
    /// let d2 = FdWeights::new(&s[..], 2);
    /// assert_eq!(d1.weights()[0], -0.5);
    /// assert_eq!(d2.weights()[0], 1.);
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
    pub fn slots(&self) -> &[isize] {
        &self.stencil.slots()
    }

    /// Returns the value of the derivative order
    pub fn ord(&self) -> usize {
        self.nderiv
    }

    /// Returns a shared reference to a rustencils::grid::ValVector
    /// that contains the calculated finite difference coefficients
    pub fn weights(&self) -> &crate::grid::ValVector {
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
    fn slots(&self) -> &[isize] {
        &self.slot_pos[..]
    }
}

#[test]
fn check_slots() {
    let slots = [3,1,0,1,-1,-2,3];
    let stncl = Stencil::new(&slots[..]);
    assert_ne!(stncl.slots(), &slots[..]);
    assert_eq!(stncl.slots(), &[-2,-1,0,1,3]);
}