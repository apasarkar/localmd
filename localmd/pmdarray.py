import numpy as np
import scipy
import scipy.sparse
from typing import *
from scipy.sparse import csr_matrix


class PMDArray():
    def __init__(self, U, R, s, V, data_shape, data_order, mean_img, std_img):
        '''
        This is a class that allows you to use the PMD output representation of the movie in an "array-like" manner. Critically, this class implements the __getitem__ function 
        which allows you to arbitrarily slice the data. To exploit other features of the PMD data (like the truncated SVD-like representation and orthogonality) mentioned below, 
        this class can serve as a starting point. Open an issue request to ask for any other demos. 
        
        Inputs: 
            U: scipy.sparse._coo.coo_matrix. Dimensions (d, K1), where K1 is larger than the estimated rank of the data. Sparse spatial basis matrix for PMD decomposition. 
            R: numpy.ndarray. Dimensions (K1, K2) where K1 >= K2. This is a mixing matrix. 
                Together: the product UR has orthonormal columns.
            s: numpy.ndarray, shape (K2,). "s" describes a diagonal matrix; we just store the diagonal values for efficiency
            V: numpy.ndarray. shape (K2, T). Has orthonormal rows. 
            data_shape: tuple of 3 ints (d1, d2, T). The first two (d1 x d2) describe the field of view dimensions and T is the number of frames
            data_order: In the compression we work with 3D data but flatten each frame into a column vector in our decomposition. This "order" param is either "F" or "C"
                and indicates how to reshape to both unflatten or flatten data. 
            mean_img: 

            Key: If you view "s" as a diagonal matrix, then (UR)s(V) is the typical representation of a truncated SVD: UR has the left singular vectors, 
                s describes the diagonal matrix, and V has the right singular vectors. We don't explicitly compute UR because U is extremely sparse, giving us 
                significantly more compression savings over large FOV data. 
        '''
        self.order = data_order
        self.d1, self.d2, self.T = data_shape
        self.U_sparse = U.tocsr()
        R = R
        s = s
        V = V
        self.V = (R * s[None, :]).dot(V) #Fewer computations
        self.mean_img = mean_img
        self.var_img = std_img
        self.row_indices = np.arange(self.d1*self.d2).reshape((self.d1, self.d2), order=self.order)
    
    @property
    def dtype(self):
        """Data type of the array elements."""
        return np.float32

    @property
    def shape(self):
        """Array dimensions."""
        return (self.T, self.d1, self.d2)

    def _parse_int_to_list(self, elt):
        if isinstance(elt, int):
            return [elt]
        else:
            return elt
    
    def __getitem__(self, key):
        """Returns self[key]."""
        if isinstance(key, int):
            implied_fov_shape = (self.d1, self.d2)
            output = U_used.dot(self.V[:, [key]]).reshape(implied_fov_shape + (-1,), order=self.order)
        elif isinstance(key, slice):
            implied_fov_shape = (self.d1, self.d2)
            output = U_used.dot(self.V[:, key]).reshape(implied_fov_shape + (-1,), order=self.order)


        elif isinstance(key, tuple):
            key = tuple([self._parse_int_to_list(key[i]) for i in range(len(key))])
            if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing columns
                U_used = self.U_sparse
                implied_fov_shape = (self.d1, self.d2)
                output = U_used.dot(self.V[:, key[0]]).reshape(implied_fov_shape + (-1,), order=self.order)
                output = output * self.var_img[(key[1], key[2], None)] + self.mean_img[(key[1], key[2], None)]
            else:
                used_rows = self.row_indices[key[1:3]]
                implied_fov_shape = used_rows.shape
                U_used = self.U_sparse[used_rows.reshape((-1,), order=self.order)]
                output = U_used.dot(self.V[:, key[0]]).reshape(implied_fov_shape + (-1,), order=self.order)
                output = output * self.var_img[(key[1], key[2], None)] + self.mean_img[(key[1], key[2], None)]

        return output.squeeze().astype(self.dtype)