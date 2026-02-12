//! Stack-allocated small linear algebra types for the Kalman filter hot loop.
//!
//! These types avoid heap allocation for small state dimensions (r <= 4)
//! that cover all practical ARMA models.

/// Stack-allocated vector of dimension `R`.
#[derive(Clone, Copy)]
pub(crate) struct SmallVec<const R: usize> {
    pub(crate) data: [f64; R],
}

/// Stack-allocated R x R matrix stored in column-major order.
///
/// `cols[c][r]` = element at row r, column c.
#[derive(Clone, Copy)]
pub(crate) struct SmallMat<const R: usize> {
    pub(crate) cols: [[f64; R]; R],
}

impl<const R: usize> SmallVec<R> {
    /// Returns a zero-initialized vector.
    #[inline(always)]
    pub(crate) fn zeros() -> Self {
        Self { data: [0.0; R] }
    }
}

impl<const R: usize> SmallMat<R> {
    /// Returns a zero-initialized matrix.
    #[inline(always)]
    pub(crate) fn zeros() -> Self {
        Self {
            cols: [[0.0; R]; R],
        }
    }

    /// Returns the element at `(row, col)`.
    #[inline(always)]
    pub(crate) fn get(&self, row: usize, col: usize) -> f64 {
        self.cols[col][row]
    }

    /// Sets the element at `(row, col)` to `val`.
    #[inline(always)]
    pub(crate) fn set(&mut self, row: usize, col: usize, val: f64) {
        self.cols[col][row] = val;
    }

    /// Computes the matrix-vector product `self * v`.
    #[inline(always)]
    pub(crate) fn mul_vec(&self, v: &SmallVec<R>) -> SmallVec<R> {
        let mut result = SmallVec::zeros();
        for i in 0..R {
            let mut sum = 0.0;
            for k in 0..R {
                sum += self.get(i, k) * v.data[k];
            }
            result.data[i] = sum;
        }
        result
    }
}
