/*
 * linsolv.c
 */

#include "linsolv.h"

#include <omp.h>
#include "setup_comm.h"
#include "exchange_matrix.h"
#include "linsys.h"
#include "matrix.h"
#include "util.h"

/*******************************************************************************
* Solves the block sparse linear system A*x0 = rhs only and directly for the
* decoupled small scale linear systems corresponding to the diagonal
* entries of A.
* Here A denotes the given block sparse matrix, rhs is the right hand
* side of the linear system and x0 is overwritten by the solution of the
* corresponding decoupled linear systems.
*
* @param[in] linsys Given linear system with LU decomposition of the diagonal
*                   entries of the block sparse matrix.
* @param[in,out] x0 On input, initial guess required for iteration.
*                   On output, solution of given linear system corresponding to
*                   the diagonal entries.
*******************************************************************************/
static void point_implicit(BSMLinSys *linsys, Matrix x0);

/*******************************************************************************
* Solves approximately the block sparse linear system A*x0 = rhs by a Jacobi
* method. Here A denotes the given block sparse matrix, rhs is the right hand
* side of the linear system and x0 is the initial guess. The maximum number
* of iterations may be defined and needs to be larger than zero.
*
* @param[in] linsys   Given linear system with LU decomposition of the diagonal
*                     entries of the block sparse matrix.
* @param[in] parallel_data Data structure required for parallelization.
* @param[in,out] x0   On input, initial guess required for iteration.
*                     On output, approximate solution of given linear system.
* @param[in] max_iter Maximum number of iterations.
*******************************************************************************/
static void jacobi(BSMLinSys *linsys, CommMap *parallel_data, Matrix x0,
                   int max_iter);

/*******************************************************************************
* Solves approximately the block sparse linear system A*x0 = rhs by a
* Gauss-Seidel method. Here A denotes the given block sparse matrix, rhs is
* the right hand side of the linear system and x0 is the initial guess. The
* maximum number of iterations may be defined and needs to be larger than zero.
*
* @param[in] linsys   Given linear system with LU decomposition of the diagonal
*                     entries of the block sparse matrix.
* @param[in] parallel_data data structure required for parallelization
* @param[in,out] x0   On input, initial guess required for iteration.
*                     On output, approximate solution of given linear system.
* @param[in] max_iter Maximum number of iterations.
*******************************************************************************/
static void gauss_seidel(BSMLinSys *linsys, CommMap *parallel_data, Matrix x0,
                         int max_iter);

/*******************************************************************************
* Solves approximately the block sparse linear system A*x0 = rhs by a symmetric
* Gauss-Seidel method. Here A denotes the given block sparse matrix, rhs is
* the right hand side of the linear system and x0 is the initial guess. The
* maximum number of iterations may be defined and needs to be larger than zero.
*
* @param[in] linsys   Given linear system with LU decomposition of the diagonal
*                     entries of the block sparse matrix.
* @param[in] parallel_data Data structure required for parallelization.
* @param[in,out] x0   On input, initial guess required for iteration.
*                     On output, approximate solution of given linear system.
* @param[in] max_iter Maximum number of iterations.
*******************************************************************************/
static void symm_gauss_seidel(BSMLinSys *linsys, CommMap *parallel_data,
                              Matrix x0, int max_iter);

/*******************************************************************************
*
*******************************************************************************/
void linsys_lu_decomp_diag(BSMLinSys *bsmls)
{
  BlockSparseMatrix A = bsmls->A;
  int **swap = bsmls->swap;

  CHECK(bsmls->decomposition_state == NOT_DECOMPOSED);
  CHECK(bsmls->diag_first);

# pragma omp parallel default(none) shared(A, swap)
  {
#   pragma omp for SCHEDULE
    for(int row = 0; row < A.num_rows; row++)
      lu_decomp(A.entries[A.row_ptr[row]], swap[row]);
  } /* end of parallel region */

  bsmls->decomposition_state = BSM_DIAG_LU_PIVOT;
} /* linsys_lu_decomp_diag() */

/*******************************************************************************
*
*******************************************************************************/
void linsolv(const BSLinSysMethod linear_solver, const int num_sweeps,
             BSMLinSys *linsys, Matrix approximate_solution, CommMap *comap)
{
  /*----------------------------------------------------------------------------
  | Depending on the choice of the iterative linear solution methodology, one
  | of the following list is chosen. All these are some kind of
  | block Jacobi or block Gauss-Seidel method.
  ----------------------------------------------------------------------------*/
  switch(linear_solver)
  {
    case BSLS_POINT_IMPLICIT:
    {
      point_implicit(linsys, approximate_solution);
      break;
    }
    case BSLS_JACOBI:
    {
      jacobi(linsys, comap, approximate_solution, num_sweeps);
      break;
    }
    case BSLS_GAUSS_SEIDEL:
    {
      gauss_seidel(linsys, comap, approximate_solution, num_sweeps);
      break;
    }
    case BSLS_SYMM_GAUSS_SEIDEL:
    {
      symm_gauss_seidel(linsys, comap, approximate_solution, num_sweeps);
      break;
    }
    default:
    {
      break;
    }
  } /* switch(linear_solver) */
} /* linsolv() */

/*******************************************************************************
*
*******************************************************************************/
static void point_implicit(BSMLinSys *linsys, Matrix x0)
{
  BlockSparseMatrix A = linsys->A;
  Matrix rhs = linsys->rhs;
  int **swap = linsys->swap;

  CHECK(linsys->decomposition_state == BSM_DIAG_LU_PIVOT);
  CHECK(linsys->diag_first);

  /*----------------------------------------------------------------------------
  | only loop over linear systems corresponding to the diagonal entries
  ----------------------------------------------------------------------------*/
# pragma omp parallel default(none) shared(A, rhs, swap, x0)
  {
#   pragma omp for SCHEDULE
    for(int row = 0; row < A.num_rows; row++)
    {
      for(int j = 0; j < rhs.cols; j++)
        x0.m[row][j] = rhs.m[row][j];

      lu_solve(A.entries[A.row_ptr[row]], x0.m[row], swap[row]);
    }
  } /* end of parallel region */
} /* point_implicit() */

/*******************************************************************************
*
*******************************************************************************/
static void jacobi(BSMLinSys *linsys, CommMap *parallel_data, Matrix x0,
                   int max_iter)
{
  BlockSparseMatrix A = linsys->A;
  Matrix rhs = linsys->rhs;
  int **swap = linsys->swap;

  Matrix help  = generateMatrix(rhs.rows, rhs.cols);

  CHECK(linsys->decomposition_state == BSM_DIAG_LU_PIVOT);
  CHECK(linsys->diag_first);

# pragma omp parallel default(none) shared(A, swap, rhs, x0, help, \
                                           parallel_data, max_iter)
  {
    int row, idx, col, i, j;
    int iter = 0;

    /*--------------------------------------------------------------------------
    | perform a Jacobi iteration
    --------------------------------------------------------------------------*/
    do
    {
      iter++;

      /* copy old solution */
#     pragma omp for SCHEDULE
      for(i = 0; i < rhs.rows; i++)
        for(j = 0; j < rhs.cols; j++)
          help.m[i][j] = x0.m[i][j];

      /* calculate new solution */
#     pragma omp for SCHEDULE
      for(row = 0; row < A.num_rows; row++)
      {
        for(i = 0; i < rhs.cols; i++)
          x0.m[row][i] = rhs.m[row][i];

        for(idx = A.row_ptr[row] + 1; idx < A.row_ptr[row + 1]; idx++)
        {
          col = A.col_index[idx];
          for(i = 0; i < A.block_size_row; i++)
            for(j = 0; j < A.block_size_col; j++)
              x0.m[row][i] -= A.entries[idx].m[i][j] * help.m[col][j];
        }

        lu_solve(A.entries[A.row_ptr[row]], x0.m[row], swap[row]);
      }

      /* communicate new solution */
#     pragma omp single
      exchange_matrix(parallel_data, x0);

    } while(iter < max_iter);
  } /* end of parallel region */

  deleteMatrix(help);
} /* jacobi() */

/*******************************************************************************
*
*******************************************************************************/
static void gauss_seidel(BSMLinSys *linsys, CommMap *parallel_data, Matrix x0,
                         int max_iter)
{
  BlockSparseMatrix A = linsys->A;
  Matrix rhs = linsys->rhs;
  int **swap = linsys->swap;
  int row, idx, col, i, j;

  int *row_ptr = A.row_ptr;
  int *col_idx = A.col_index;
  const int brows = A.block_size_row;
  const int bcols = A.block_size_col;
  const int num_rows = A.num_rows;

  CHECK(linsys->decomposition_state == BSM_DIAG_LU_PIVOT);
  CHECK(linsys->diag_first);

# pragma omp parallel default(none) shared(A, col_idx, row_ptr, rhs, \
                                           parallel_data, swap, max_iter, x0) \
                                     private(row, idx, col, i, j)
  {
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int iter = 0;

    CHECK(nthreads > 0);

    int chunkmin = num_rows / nthreads;
    int rest = num_rows % nthreads;
    int mychunksize = chunkmin + (threadid < rest);
    int start = threadid * chunkmin + MIN(threadid, rest);
    int stop = start + mychunksize;

    Matrix my_x0 = generateMatrix(mychunksize, rhs.cols);
    double *x0_row = NULL;

    /*--------------------------------------------------------------------------
    | perform a Gauss-Seidel iteration
    --------------------------------------------------------------------------*/
    do
    {
      iter++;

      /* each thread: copy own chunk of current local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          my_x0.m[i - start][j] = x0.m[i][j];

      /* each thread: calculate own chunk of new solution */
      for(row = start; row < stop; row++)
      {
        const int myrow = row - start;

        for(i = 0; i < rhs.cols; i++)
          my_x0.m[myrow][i] = rhs.m[row][i];

        for(idx = row_ptr[row] + 1; idx < row_ptr[row + 1]; idx++)
        {
          col = col_idx[idx];

          if(col >= start && col < stop)
            x0_row = my_x0.m[col - start]; /* use own new solution */
          else
            x0_row = x0.m[col]; /* use old solution */

          for(i = 0; i < brows; i++)
            for(j = 0; j < bcols; j++)
              my_x0.m[myrow][i] -= A.entries[idx].m[i][j] * x0_row[j];
        }

        lu_solve(A.entries[row_ptr[row]], my_x0.m[myrow], swap[row]);
      }

#     pragma omp barrier
      /* each thread: update own chunk of local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          x0.m[i][j] = my_x0.m[i - start][j];

#     pragma omp barrier
      /* communicate new solution */
#     pragma omp single
      exchange_matrix(parallel_data, x0);

    } while(iter < max_iter);

    deleteMatrix(my_x0);
  } /* end of parallel region */
} /* gauss_seidel() */

/*******************************************************************************
*
*******************************************************************************/
static void symm_gauss_seidel(BSMLinSys *linsys, CommMap *parallel_data,
                              Matrix x0, int max_iter)
{
  BlockSparseMatrix A = linsys->A;
  Matrix rhs = linsys->rhs;
  int **swap = linsys->swap;
  int row, idx, col, i, j;

  int *row_ptr = A.row_ptr;
  int *col_idx = A.col_index;
  const int brows = A.block_size_row;
  const int bcols = A.block_size_col;
  const int num_rows = A.num_rows;

  CHECK(linsys->decomposition_state == BSM_DIAG_LU_PIVOT);
  CHECK(linsys->diag_first);

# pragma omp parallel default(none) shared(A, col_idx, row_ptr, rhs, \
                                           parallel_data, swap, max_iter, x0) \
                                     private(row, idx, col, i, j)
  {
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int iter = 0;

    CHECK(nthreads > 0);

    int chunkmin = num_rows / nthreads;
    int rest = num_rows % nthreads;
    int mychunksize = chunkmin + (threadid < rest);
    int start = threadid * chunkmin + MIN(threadid, rest);
    int stop = start + mychunksize;

    Matrix my_x0 = generateMatrix(mychunksize, rhs.cols);
    double *x0_row = NULL;

    /*--------------------------------------------------------------------------
    | perform a symmetric Gauss-Seidel iteration
    --------------------------------------------------------------------------*/
    do
    {
      iter++;

      /*------------------------------------------------------------------------
      | Forward sweep
      ------------------------------------------------------------------------*/
      /* each thread: copy own chunk of current local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          my_x0.m[i - start][j] = x0.m[i][j];

      /* each thread: calculate own chunk of new solution start...stop-1 */
      for(row = start; row < stop; row++)
      {
        const int myrow = row - start;

        for(i = 0; i < rhs.cols; i++)
          my_x0.m[myrow][i] = rhs.m[row][i];

        for(idx = row_ptr[row] + 1; idx < row_ptr[row + 1]; idx++)
        {
          col = col_idx[idx];

          if(col >= start && col < stop)
            x0_row = my_x0.m[col - start]; /* use own new solution */
          else
            x0_row = x0.m[col]; /* use old solution */

          for(i = 0; i < brows; i++)
            for(j = 0; j < bcols; j++)
              my_x0.m[myrow][i] -= A.entries[idx].m[i][j] * x0_row[j];
        }

        lu_solve(A.entries[row_ptr[row]], my_x0.m[myrow], swap[row]);
      }

#     pragma omp barrier
      /* each thread: update own chunk of local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          x0.m[i][j] = my_x0.m[i - start][j];

#     pragma omp barrier
      /* communicate new solution */
#     pragma omp single
      exchange_matrix(parallel_data, x0);

      /*------------------------------------------------------------------------
      | Backward sweep
      ------------------------------------------------------------------------*/
      /* each thread: copy own chunk of current local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          my_x0.m[i - start][j] = x0.m[i][j];

      /* each thread: calculate own chunk of new solution stop-1...start */
      for(row = stop - 1; row >= start; row--)
      {
        const int myrow = row - start;

        for(i = 0; i < rhs.cols; i++)
          my_x0.m[myrow][i] = rhs.m[row][i];

        for(idx = row_ptr[row] + 1; idx < row_ptr[row + 1]; idx++)
        {
          col = col_idx[idx];

          if(col >= start && col < stop)
            x0_row = my_x0.m[col - start]; /* use own new solution */
          else
            x0_row = x0.m[col]; /* use old solution */

          for(i = 0; i < brows; i++)
            for(j = 0; j < bcols; j++)
              my_x0.m[myrow][i] -= A.entries[idx].m[i][j] * x0_row[j];
        }

        lu_solve(A.entries[row_ptr[row]], my_x0.m[myrow], swap[row]);
      }

#     pragma omp barrier
      /* each thread: update own chunk of local solution */
      for(i = start; i < stop; i++)
        for(j = 0; j < rhs.cols; j++)
          x0.m[i][j] = my_x0.m[i - start][j];

#     pragma omp barrier
      /* communicate new solution */
#     pragma omp single
      exchange_matrix(parallel_data, x0);

    } while(iter < max_iter);

    deleteMatrix(my_x0);
  } /* end of parallel region */
} /* symm_gauss_seidel() */
