import numpy as np
import scipy.sparse
from typing import *
from scipy.sparse import coo_matrix


class PMDArray:
    def __init__(self, U: coo_matrix, R: np.ndarray, s: np.ndarray,
                 V: np.ndarray, data_shape: tuple[int, int, int], data_order: str, mean_img: np.ndarray,
                 std_img: np.ndarray):
        """
        This is a class that allows you to use the PMD output representation of the movie in an "array-like" manner.
        Critically, this class implements the __getitem__ functionality allowing you to index the data, etc.
        which allows you to efficiently slice frames and spatial subsets of the data.
        
        Args:
            U (scipy.sparse._coo.coo_matrix): Dimensions (d, K1), where K1 is larger than the estimated rank of the data.
                Sparse spatial basis matrix for PMD decomposition.
            R (numpy.ndarray): Dimensions (K1, K2) where K1 >= K2. This is a mixing matrix.
                Together: the product UR has orthonormal columns.
            s (numpy.ndarray): Shape (K2,). "s" describes a diagonal matrix; we just store the
                diagonal values for efficiency
            V (numpy.ndarray): shape (K2, T). Has orthonormal rows.
            data_shape (tuple): Tuple of 3 ints (T, d1, d2). The first two (d1 x d2) describe the field of view
                dimensions and T is the number of frames
            data_order: In the compression we work with 3D data but flatten each frame into a column vector in our
                decomposition. This "order" param is either "F" or "C" and indicates how to reshape to both
                unflatten or flatten data.
            mean_img (np.ndarray): Shape (d1, d2). Pixel-wise mean image of data
            std_img (np.ndarray): Shape (d1, d2). Noise variance image of the data

            Key: If you view "s" as a diagonal matrix, then (UR)s(V) is the typical representation of a truncated SVD:
                UR has the left singular vectors, s describes the diagonal matrix, and V has the right singular vectors.
                We don't explicitly compute UR because U is extremely sparse, giving us
                significantly more compression savings over large FOV data.
        """
        self.order = data_order
        self.T, self.d1, self.d2 = data_shape
        self.U_sparse = U.tocsr()
        self.R = R
        self.s = s
        self.V = V
        self._RsV = (R * s[None, :]).dot(V)  # Fewer computations when doing __getitem__
        self.mean_img = mean_img
        self.var_img = std_img
        self.row_indices = np.arange(self.d1 * self.d2).reshape((self.d1, self.d2), order=self.order)

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
        return 3

    def _parse_int_to_list(self, elt):
        if isinstance(elt, int):
            return [elt]
        else:
            return elt

    def spatial_crop(self, key) -> tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray, tuple]:
        """

        Args:
            key (tuple): Length 2 tuple used to slice the rows of the data
        Returns:
            U_used (scipy.sparse.csr_matrix). Cropped sparse spatial matrix
            mean_used (np.ndarray):
            implied_fov (tuple). Tuple of integer(s) specifying the implied FOV dimensions
        """
        if key[0] is None or key[1] is None:
            raise ValueError("Cannot pass in None for indexing")

        key = (self._parse_int_to_list(key[0]), self._parse_int_to_list(key[1]))
        used_rows = self.row_indices[key[0], key[1]]
        mean_used = self.mean_img[key[0], key[1]]
        var_used = self.var_img[key[0], key[1]]
        u_used = self.U_sparse[used_rows.reshape((-1,), order=self.order)]
        implied_fov_shape = used_rows.shape
        return u_used, mean_used, var_used, implied_fov_shape

    def temporal_crop(self, key: Union[np.ndarray, slice, list, int]) -> np.ndarray:
        """

        Args:
            key (array, list, int): Key used for temporal crop

        Returns:
        """
        if key is None:
            raise ValueError("Cannot use None for indexing")

        return self._RsV[:, self._parse_int_to_list(key)]

    def __getitem__(self, key) -> np.ndarray:
        """Returns self[key]. Does NOT support dimension expansion."""
        if key is None:
            raise ValueError("Cannot use None for indexing")

        if not isinstance(key, tuple):
            key = (key,)

        if len(key) == 1:
            spatial, mean_used, var_used, implied_fov_dims = self.spatial_crop(
                (slice(None, None, None), slice(None, None, None)))
            temporal = self.temporal_crop(key[0])
        elif len(key) == 2:
            spatial, mean_used, var_used, implied_fov_dims = self.spatial_crop(key[1], slice(None, None, None))
            temporal = self.temporal_crop(key[0])
        elif len(key) == 3:
            spatial, mean_used, var_used, implied_fov_dims = self.spatial_crop((key[1], key[2]))
            temporal = self.temporal_crop(key[0])
        else:
            raise ValueError("Too many values to unpack in __getitem__")

        # Get the unnormalized outputs
        output = spatial.dot(temporal)
        output = (output.reshape(implied_fov_dims + (-1,), order=self.order) *
                  np.expand_dims(var_used, axis=len(var_used.shape)) +
                  np.expand_dims(mean_used, axis=len(mean_used.shape)))

        # Return with the frames as the first dimension
        output = np.transpose(output, axes=(len(output.shape) - 1, *range(len(output.shape) - 1)))

        return output.squeeze().astype(self.dtype)
