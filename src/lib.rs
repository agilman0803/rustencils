#![crate_name = "rustencils"]
pub mod grid;
pub mod stencil;
pub mod operator;

pub mod problem {
    pub mod heat_eqn {
        use crate::grid::{GridSpec, GridScalar, ValVector};
        use crate::operator;

        pub fn from_default_laplacian(alpha: f64, qty: &GridScalar) -> impl (Fn(GridScalar) -> ValVector)
                // F: Fn(Q) -> Q
        {
            let f = { |q: GridScalar| {q.gridvals().clone()} };
            f
        }
    }
}

pub mod dirver {
    /// The SpatialDriver trait defines an object that is responsible for
    /// 
    pub trait SpatialDriver {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn feels_good() {
        // Want users to be able to write code that looks like the following:
        // use std::collections::HashMap;
        // use std::rc::Rc;
        use crate::{grid, stencil, operator, problem};

        let t_start = 0.;
        let dt = 0.001;
        let t_end = 0.1;

        let mut axs = grid::GridSpace::new();
        axs.add_axis('x', 0., 0.01, 50, false);
        let mut spec = grid::CartesianGridSpec::new(axs);
        grid::GridScalar::zeros(&mut spec, 'T');
        spec.scalars_mut()[&'T'].add_bound_condition(grid::DirichletConstant::new(1., &spec, 'x', grid::BoundarySide::Low));
        spec.scalars_mut()[&'T'].add_bound_condition(grid::DirichletConstant::new(0., &spec, 'x', grid::BoundarySide::High));

        let heat_eqn = problem::heat_eqn::from_default_laplacian(0.05, &spec.scalars()[&'T']);
        // let heat_eqn = |qty: grid::GridScalar<grid::CartesianGridSpec>| { ... };

        spec.add_function(heat_eqn, 'T');
    }
}

/*
The order of operations for one cycle of the solver is:
1. The boundary points are set to appropriate values.
2. The time and values are logged.
3. The change in the values is calculated according to
   the desired algorithm (e.g., Forward Euler, RK4) and
   the RHS function.
    (i.e. dy/dt = RHS which does not involve d/dt but
    can involve t)
    RHS is function of dependent and independent
    variables.
    Note: Is it possible to make LHS d/dn where n is
          iteration count? This would be useful to
          solve general optimization problems.
4. The dependent variables are updated to new values
   according to the change calculated in step 3.
5. The time/iteration count is incremented.
*/

/*
For coordinate systems that involve radial and angular components:

Pass in a minimum (greatest radial component) and maximum (lowest
radial component) angular spacing, along with the number of radial
points where r > 0. (Maybe just have manually set max and/or min 
angular spacing?) At each concentric radial step, the angular spacing
is multiplied by a constant.
*/

/*
Iterating on the RHS interface:
*/
fn rhs_eval<S: grid::GridSpec, F: Fn(grid::GridScalar)->grid::ValVector>(spec: &S, rhs: &F) {}

fn heat_maybe<S: grid::GridSpec>(tmptr_label: char) -> impl (Fn(f64, &S)->grid::ValVector) {
    |time: f64, spec: &S| {
        spec.scalars()[&tmptr_label].gridvals().clone()
    }
}
/*
How will setting boundaries work?
Want some outer object on which to simply call:

_outer_obj.set_bounds(_grid_qty);

So outer object must already know what the conditions are for each axis.

Alternatively, there could be a boundary driver that takes both the grid
qty and a vector of boundary conditions and just iterates over the vector,
setting each boundary as it goes. The driver could be an object or
standalone function.

For a Dirichlet BC, there are predicted to be four general cases that
will require specialized implementations:
1. The value at every boundary point is fixed at one constant value
    independent of the timestep (may still depend on position).
2. The value at every boundary point is dependent on the timestep
    in a "random" way (i.e., the dependence can't be expressed as
    a function). The values may of course also depend on position.
    This may actually be difficult since there may not be a good way
    to associate a Vec of values with a specific timestep. Could
    simply have a Vec<Vec<(idx: usize, value: f64)>> that is slowly
    consumed over the course of the simulation.
3. The value at every boundary point is dependent on the timestep
    (and position) in a way that can be expressed as a mathematical
    function, which itself does not depend on the timestep.
4. (Is this possible?) The value at every boundary point is dependent
    on the timestep (and position) in a way that can be expressed as
    a mathematical function, which itself DOES depend on the timestep.

Maybe each boundary condition needs a method to generate the boundary
values in some standardized way (e.g. Vec<(idx: usize, value: f64)>).

In realtiy, due to the inability to know how the grid will be constructed
at compile time, the dependence of the boundary value on position must be
an analytical expression to be computed at run time. This combined with
the complexity of comparing/hashing floating point values means that the
only options which are straightforward to implement are the following:
1. The value at the boundary is constant with respect to space and time.
2. The value at the boundary is an analytical function f(t, x0,...,xn).
    This function may be piece-wise since ordering of floating point
    values is allowed.
3. The value at the boundary is constant with respect to space, and the
    user assumes responsibility for the time dependence by passing in a
    Vec of values that are consumed one at a time at each timestep.
4. The value at the boundary is an analytical function g(t, x0,...,xn) which
    may be piece-wise, and the user assumes responsibility for the time
    dependence by passing in a Vec of functions (or function pointers?)
    that are consumed one at a time at each timestep.

Therefore, perhaps the most logical construction is for a boundary driver
object to hold a vector of objects that each implement a BoundaryCondition
trait that only has one method which is called to generate the vector of
boundary values. Then the driver can iterate over the values and set them
in the grid qty as it goes.
*/