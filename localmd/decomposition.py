import jax
import jax.scipy
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial

import jaxopt
import numpy as np

from localmd.evaluation import spatial_roughness_stat_vmap, temporal_roughness_stat_vmap, construct_final_fitness_decision
from localmd.preprocessing_utils import standardize_block

@partial(jit)
def objective_function(X, placeholder, data):
    num_rows = data.shape[0]
    difference = data.shape[1]

    comp1 = jax.lax.dynamic_slice(X, (0, 0), (num_rows, X.shape[1]))
    comp2 = jax.lax.dynamic_slice(X, (num_rows, 0), (difference, X.shape[1]))
    prod = jnp.matmul(comp1, comp2.T)
    return jnp.linalg.norm(data - prod)

#Some observations here: the projected gradient step is significantly slower than unconstrained gradient 
# and it not necessary for us to actually use 
@partial(jit)
def rank_k_fit(data, orig_placeholder):

    shape_1 = data.shape[0]
    shape_2 = data.shape[1]
    init_param = jnp.zeros((shape_1 + shape_2, orig_placeholder.shape[0])) + 1
    solver = ProjectedGradient(fun=objective_function,
                             projection=projection.projection_non_negative,
                             tol=1e-6, maxiter=1000)
    fit_val = solver.run(init_param, placeholder=orig_placeholder, data=data).params

    return fit_val


rank_k_fit_vmap = jit(vmap(rank_k_fit, in_axes=(2, None)))



@partial(jit)
def unconstrained_rank_fit(data, orig_placeholder):

    shape_1 = data.shape[0]
    shape_2 = data.shape[1]
    init_param = jnp.zeros((shape_1 + shape_2, orig_placeholder.shape[0])) + 1

    solver = jaxopt.GradientDescent(fun=objective_function, maxiter=1000)
    params, state = solver.run(init_param, placeholder=orig_placeholder, data=data)

    return params

@partial(jit)
def add_ith_column(mat, vector, i):
    '''
    Jit-able function for adding a vector to a specific column of a matrix (i-th column)
    Inputs: 
        mat: jnp.array, size (d, T)
        vector: jnp.array, size (1, T)
        i: integer between 0 and T-1 inclusive. Denotes which column of mat you'd like to add "vector" to
        
    Returns: 
        mat: jnp.array, size (d, T). 
    '''
    col_range = jnp.arange(mat.shape[1])
    col_filter = col_range == i
    col_filter = jnp.expand_dims(col_filter, axis=0)
    
    dummy_mat = jnp.zeros_like(mat)
    dummy_mat = dummy_mat + vector 
    dummy_mat = dummy_mat * col_filter
    mat = mat + dummy_mat
    
    return mat
    
    
@partial(jit)
def rank_1_deflation_pytree(i, input_pytree):
    '''
    Computes a rank-1 decomposition of residual and appends the results to u and v
    Inputs:
        i: integer indicating which column of our existing data to update with our rank-1 decomposition
            (this is clarified below, look at the parameter definitions for u and v)
        input_pytree. a list python object containing the following jnp.arrays (in order):
            residual: jnp.array. Shape (d, T)
            u: jnp.array. Shape (d, K) for some K that is not relevant to this functionality. We append the column vector of
            our rank-1 decomposition to column k of u.
            v: jnp.array. Shape (K, T). We append the row vector of
                our rank-1 decomposition to row k of v.
            
    Outputs:
        cumulative_results. Pytree (python list) containing (residual, u, v), where the residual has been updated
            (i.e. we subtracted the rank-1 estimate from this procedure) and u and v have been updated as well 
                (i.e. we append the rank-1 estimate from this procedure to the i-th column/row of u/v respectively).

    
    '''
    residual = input_pytree[0]
    u = input_pytree[1]
    v = input_pytree[2]
    
    #Step 1: Get a rank-1 fit for the data
    placeholder = jnp.zeros((1, 1))
    approximation = unconstrained_rank_fit(residual, placeholder) #Output here will be (d + T, 1)-shaped jnp array
    u_k = jax.lax.dynamic_slice(approximation, (0,0), (residual.shape[0], 1))
    v_k = jax.lax.dynamic_slice(approximation, (residual.shape[0], 0), (residual.shape[1], 1))
    
    # v_k = jnp.dot(residual.T, u_k) #This is the debias/rescaling step
    
    new_residual = residual - jnp.dot(u_k, v_k.T)
    
    u_new = add_ith_column(u, u_k, i)
    v_new = add_ith_column(v.T, v_k, i).T
    
    return [new_residual, u_new, v_new]    
    
@partial(jit)
def iterative_rank_1_approx(test_data):
    num_iters = 25
    u_mat = jnp.zeros((test_data.shape[0], num_iters))
    v_mat = jnp.zeros((num_iters, test_data.shape[1]))
    i = 0
    data_pytree = [test_data, u_mat, v_mat]
    final_pytree = jax.lax.fori_loop(0, num_iters, rank_1_deflation_pytree, data_pytree)
    
    return final_pytree

@partial(jit)
def iterative_rank_1_approx_sims(test_data):
    num_iters = 3
    u_mat = jnp.zeros((test_data.shape[0], num_iters))
    v_mat = jnp.zeros((num_iters, test_data.shape[1]))
    i = 0
    data_pytree = [test_data, u_mat, v_mat]
    final_pytree = jax.lax.fori_loop(0, num_iters, rank_1_deflation_pytree, data_pytree)
    
    return final_pytree



@partial(jit)
def decomposition_no_normalize(block):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    decomposition = iterative_rank_1_approx_sims(block_2d)
    
    u_mat, v_mat = decomposition[1], decomposition[2]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics

decomposition_no_normalize_vmap = jit(vmap(decomposition_no_normalize, in_axes = (3)))

def threshold_heuristic(block_sizes, iters=10, num_sims=5):
    '''
    We simulate the roughness statistics for components when we fit to standard normal noise
    Inputs: 
        block_sizes: tuple, dimensions of block: d1, d2, T, where d1 and d2 are the dimensions of the block on the FOV and T is the window length
        iters: default parameter, int. Number of times we do this procedure on the GPU. This param is really to avoid memorry blowups on the GPU
        num_sims: default parameter, int. 
    Outputs: 
        spatial_thresh, temporal_thresh. The spatial and temporal statistics
    '''
    d1, d2, T = block_sizes
    spatial_cumulator = np.zeros((0,))
    temporal_cumulator = np.zeros((0, ))

    for j in range(iters):
        noise_data = np.random.randn(d1, d2, T, num_sims)

        results = decomposition_no_normalize_vmap(noise_data)

        spatial_temp = results[0].reshape((-1,))
        temporal_temp = results[1].reshape((-1,))

        spatial_cumulator = np.concatenate([spatial_cumulator, spatial_temp])
        temporal_cumulator = np.concatenate([temporal_cumulator, temporal_temp])

    spatial_thres = np.percentile(spatial_cumulator.flatten(), 5)
    temporal_thres = np.percentile(temporal_cumulator.flatten(), 5)
    
    return spatial_thres, temporal_thres




@partial(jit)
def single_block_md(block, spatial_thres, temporal_thres, max_consec_failures):
    block = standardize_block(block) #Center and divide by noise standard deviation before doing matrix decomposition
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    
    
    
    decomposition = iterative_rank_1_approx(block_2d)
    u_mat, v_mat = decomposition[1], decomposition[2]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    

    
    ##Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres, max_consec_failures)
    
    u_mat = jnp.reshape(u_mat, (d1*d2, -1), order="F")
    
    good_comps_expanded = jnp.expand_dims(good_comps, axis=0)
    u_mat_filtered = u_mat * good_comps_expanded
    Q, R = jnp.linalg.qr(u_mat_filtered, mode="reduced")
    return jnp.reshape(Q, (d1, d2, -1), order="F")

@partial(jit)
def single_block_md_new(block, spatial_thres, temporal_thres, max_consec_failures):
    #TODO: Get rid of max consec failures entirely from function API 
    block = standardize_block(block) #Center and divide by noise standard deviation before doing matrix decomposition
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    
    
    
    decomposition = iterative_rank_1_approx(block_2d)
    u_mat, v_mat = decomposition[1], decomposition[2]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    

    
    ##Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres,\
                                                  temporal_thres, max_consec_failures)
    
    u_mat = jnp.reshape(u_mat, (d1*d2, -1), order="F")
    return u_mat, good_comps

single_block_md_new_vmap = jit(vmap(single_block_md_new, in_axes=(3, None, None, None)))