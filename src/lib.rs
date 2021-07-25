#![crate_name = "rustencils"]
pub mod grid;
pub mod stencil;
pub mod operator;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
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