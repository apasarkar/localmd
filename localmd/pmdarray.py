import numpy as np
import scipy
import scipy.sparse
from typing import *
from scipy.sparse import csr_matrix
from localmd.lazy_array import lazy_data_loader


class PMDArray(lazy_data_loader):
    def __init__(self, U: csr_matrix, R: np.ndarray, s: np.ndarray,
                 V: np.ndarray, data_shape: tuple[int, int, int], data_order: str, mean_img: np.ndarray,
                 std_img: np.ndarray) -> None:
        """
        This is a class that allows you to use the PMD output representation of the movie in an "array-like" manner. Critically, this class implements the __getitem__ function
        which allows you to arbitrarily slice the data. To exploit other features of the PMD data (like the truncated SVD-like representation and orthogonality) mentioned below,
        this class can serve as a starting point. Open an issue request to ask for any other demos.

        Inputs:
            U (scipy.sparse.csr_matrix): Dimensions (d, K1), where K1 is larger than the estimated rank of the data.
                Sparse spatial basis matrix for PMD decomposition.
            R (numpy.ndarray): Dimensions (K1, K2) where K1 >= K2. This is a mixing matrix.
                Together: the product UR has orthonormal columns.
            s (numpy.ndarray): shape (K2,). "s" describes a diagonal matrix; we just store the diagonal values for efficiency
            V (numpy.ndarray): shape (K2, T). Has orthonormal rows.
            data_shape (tuple): of 3 ints (d1, d2, T). The first two (d1 x d2) describe the field of view dimensions and T is the number of frames
            data_order (str): In the compression we work with 3D data but flatten each frame into a column vector in our decomposition. This "order" param is either "F" or "C"
                and indicates how to reshape to both unflatten or flatten data.
            mean_img (np.ndarray): Shape (d1, d2). Pixelwise mean image.
            std_img (np.ndarray): Shape (d1, d2). Pixelwise noise variance image.

        Key: If you view "s" as a diagonal matrix, then (UR)s(V) is the typical representation of a truncated SVD: UR has the left singular vectors,
            s describes the diagonal matrix, and V has the right singular vectors. We don't explicitly compute UR because U is extremely sparse, giving us
            significantly more compression savings over large FOV data.
        """
        self.order = data_order
        self.d1, self.d2, self.T = data_shape
        self.U_sparse = U.astype('float32')
        self.R = R.astype(self.dtype)
        self.s = s.astype(self.dtype)
        self.V = V.astype(self.dtype)
        self.mean_img = mean_img.astype(self.dtype)
        self.var_img = std_img.astype(self.dtype)

    @property
    def dtype(self):
        """Data type of the array elements."""
        return np.float32

    @property
    def shape(self):
        """Array dimensions."""
        return (self.T, self.d1, self.d2)

    @property
    def ndim(self):
        """ Number of dimensions """
        return len(self.shape)

    @property
    def n_frames(self):
        return self.T

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Args:
            indices (int or slice): For selecting frames of the data

        Returns:
            output (np.ndarray): (T, d1, d2)-shaped np.ndarray, where (d1, d2) are the FOV dimensions
        """
        if isinstance(indices, int):
            V_used = [indices]
        else:
            V_used = indices
        right_mat = self.R.dot(self.s[:, None] * self.V[:, V_used])
        output = self.U_sparse.dot(right_mat)
        output = output.reshape((self.d1, self.d2, -1), order=self.order)
        return output.transpose(2, 0, 1).squeeze()