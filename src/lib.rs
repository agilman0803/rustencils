pub mod grid {
    // extern crate ndarray;
    // use ndarray::Array;

    pub trait GridSpec {
        fn get_coords(&self) -> &Vec<Vec<f32>>;
        fn get_ndim(&self) -> usize;
        fn get_gridshape(&self) -> &Vec<usize>;
        fn get_spacing(&self) -> &Vec<f32>;
    }

    pub struct CartesianGridSpec {
        coords: Vec<Vec<f32>>, // e.g. [[x0,x1,x2,x3,x4,...,xm],[y0,y1,y2,y3,y4,...,yn]]
        ndim: usize,           // e.g. 2
        gridshape: Vec<usize>, // e.g. [m,n]
        spacing: Vec<f32>,     // e.g. [(x1-x0),(y1-y0)]
    }

    impl GridSpec for CartesianGridSpec {
        fn get_coords(&self) -> &Vec<Vec<f32>> { &self.coords }
        fn get_ndim(&self) -> usize { self.ndim }
        fn get_gridshape(&self) -> &Vec<usize> { &self.gridshape }
        fn get_spacing(&self) -> &Vec<f32> { &self.spacing }
    }

    impl CartesianGridSpec {
        pub fn new(coords: Vec<Vec<f32>>) -> Self {
            CartesianGridSpec {
                spacing: Self::init_spacing(&coords),
                gridshape: Self::init_gridshape(&coords),
                ndim: coords.len(),
                coords: coords,
            }
        }

        fn init_gridshape(coords: &Vec<Vec<f32>>) -> Vec<usize> {
            let mut gridshape = Vec::with_capacity(coords.len());
            for set in coords.iter() {
                gridshape.push(set.len());
            }

            gridshape
        }
        
        fn init_spacing(coords: &Vec<Vec<f32>>) -> Vec<f32> {
            let mut spacing = Vec::with_capacity(coords.len());
            for set in coords.iter() {
                spacing.push(set[1]-set[0]);
            }

            spacing
        }
    }

    pub trait GridQty {
        type Spec: GridSpec;
        fn get_spec(&self) -> &Self::Spec;
        fn get_qtyshape(&self) -> &Vec<usize>;
        fn get_gridvals(&self) -> &Vec<Vec<f64>>;
        fn get_grididxs(&self) -> &Vec<Vec<Vec<i32>>>;
    }

    // for something like a velocity field with components [vx, vy], the `gridvals` vector should look like:
    // [[vx0, vy0],
    //  [vx1, vy1],
    //  [vx2, vy2],
    //  [vx3, vy3],
    //  [vx4, vy4],
    //  ...,
    //  [vxn, vyn]]
    //
    // and the `gridindices` would look something like:
    // [ [[0], [1], [2]],
    //   [[3], [4], [5]],
    //   [[6], [7], [8]] ]

    pub struct GridScalarVector<T: GridSpec> {
        spec: T,
        qtyshape: Vec<usize>, // The shape of the values at each point on the grid: vec![1] (scalar) or vec![3] (3-component vector)
        gridvals: Vec<Vec<f64>>, // A vector that just contains the values of interest at every point
        grididxs: Vec<Vec<Vec<i32>>>, // A vector of vectors representing 3D space. Indices should be [x][y][z].
                                         // Each innermost element is an index of the `gridvals` vector.
    }

    impl<T> GridQty for GridScalarVector<T> where T: GridSpec {
        type Spec = T;
        fn get_spec(&self) -> &Self::Spec { &self.spec }
        fn get_qtyshape(&self) -> &Vec<usize> { &self.qtyshape }
        fn get_gridvals(&self) -> &Vec<Vec<f64>> { &self.gridvals }
        fn get_grididxs(&self) -> &Vec<Vec<Vec<i32>>> { &self.grididxs }
    }

    impl<T> GridScalarVector<T> where T: GridSpec {
        pub fn new(spec: T, grididxs: Vec<Vec<Vec<i32>>>, gridvals: Vec<Vec<f64>>) -> Self {
            GridScalarVector {
                spec: spec,
                qtyshape: Self::init_qtyshape(&gridvals),
                gridvals: gridvals,
                grididxs: grididxs,
            }
        }

        fn init_qtyshape(gridvals: &Vec<Vec<f64>>) -> Vec<usize> {
            vec![gridvals[0].len()]
        }

        // pub fn apply_operator(self, operator: ?) -> Self { todo!() }
    }

    impl<T> std::ops::Add for GridScalarVector<T> where T: GridSpec {
        type Output = Self;

        fn add(self, other: GridScalarVector<T>) -> Self {
            if self.gridvals.len() != other.gridvals.len() { panic!("Can't add two grids of different sizes!") }
            if self.gridvals[0].len() != other.gridvals[0].len() { panic!("Grid elements must be the same size!") }
            if self.grididxs != other.grididxs { panic!("Grid indexes must be the same!") }

            let mut result = vec![vec![0.0; self.gridvals[0].len()]; self.gridvals.len()];
            for ((rout, sout), oout) in result.iter_mut().zip(&self.gridvals).zip(&other.gridvals) {
                for ((r, s), o) in rout.iter_mut().zip(sout).zip(oout) {
                    *r = s + o;
                }
            }
            GridScalarVector {
                spec: self.spec,
                qtyshape: self.qtyshape,
                gridvals: result,
                grididxs: self.grididxs,
            }
        }
    }

    impl<T> std::ops::Add<f64> for GridScalarVector<T> where T: GridSpec {
        type Output = Self;

        fn add(self, other: f64) -> Self {
            let mut result = vec![vec![0.0; self.gridvals[0].len()]; self.gridvals.len()];
            for (rout, sout) in result.iter_mut().zip(&self.gridvals) {
                for (r, s) in rout.iter_mut().zip(sout) {
                    *r = s + other;
                }
            }
            GridScalarVector {
                spec: self.spec,
                qtyshape: self.qtyshape,
                gridvals: result,
                grididxs: self.grididxs,
            }
        }
    }

    impl<T> std::ops::Mul for GridScalarVector<T> where T: GridSpec {
        type Output = Self;

        fn mul(self, other: GridScalarVector<T>) -> Self {
            if self.gridvals.len() != other.gridvals.len() { panic!("Can't add two grids of different sizes!") }
            if self.gridvals[0].len() != other.gridvals[0].len() { panic!("Grid elements must be the same size!") }
            if self.grididxs != other.grididxs { panic!("Grid indexes must be the same!") }

            let mut result = vec![vec![0.0; self.gridvals[0].len()]; self.gridvals.len()];
            for ((rout, sout), oout) in result.iter_mut().zip(&self.gridvals).zip(&other.gridvals) {
                for ((r, s), o) in rout.iter_mut().zip(sout).zip(oout) {
                    *r = s * o;
                }
            }
            GridScalarVector {
                spec: self.spec,
                qtyshape: self.qtyshape,
                gridvals: result,
                grididxs: self.grididxs,
            }
        }
    }

    impl<T> std::ops::Mul<f64> for GridScalarVector<T> where T: GridSpec {
        type Output = Self;

        fn mul(self, other: f64) -> Self {
            let mut result = vec![vec![0.0; self.gridvals[0].len()]; self.gridvals.len()];
            for (rout, sout) in result.iter_mut().zip(&self.gridvals) {
                for (r, s) in rout.iter_mut().zip(sout) {
                    *r = s * other;
                }
            }
            GridScalarVector {
                spec: self.spec,
                qtyshape: self.qtyshape,
                gridvals: result,
                grididxs: self.grididxs,
            }
        }
    }
}

pub mod operator {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
