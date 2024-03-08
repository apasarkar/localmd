import numpy as np
import tifffile
from abc import ABC, abstractmethod
from typing import *

class lazy_data_loader(ABC):
    '''
    This captures the numpy array-like functionality that all data loaders for motion correction need to contain

    Key: To implement support for a new file type, you just need to specify the key properties below (dtype, shape, ndim)
    and then implement the function _compute_at_indices.
    Adapted from mesmerize core: https://github.com/nel-lab/mesmerize-core/blob/master/mesmerize_core/arrays/_base.py
    '''

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        pass

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def __getitem__(
            self,
            item: Union[int, list, Tuple[Union[int, slice, range]]]
    ):
        if isinstance(item, list):
            return self._compute_at_indices(item)

        elif isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item
        
        elif isinstance(item, np.ndarray):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]




        else:
            raise IndexError(
                f"Invalid indexing method, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame start index of <{start}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")
            if stop is not None:
                if stop > self.shape[0]:
                    raise IndexError(f"Cannot index beyond `n_frames`.\n"
                                     f"Desired frame stop index of <{stop}> "
                                     f"lies beyond `n_frames` <{self.shape[0]}>")

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    @abstractmethod
    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here to return frames. Slices the array over time (dimension 0) at the desired indices.

        Parameters
        ----------
        indices: Union[list, int, slice]
            the user's desired way of picking frames, either an int, list of ints, or slice
             i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        pass    

class TiffArray(lazy_data_loader):

    def __init__(self, filename):
        """
        TiffArray data loading object. Supports loading data from multipage tiff files.

        Args:
            filename (str): Path to file

        """
        self.filename = filename

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
            x, y = page.shape
        return num_frames, x, y

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        if isinstance(indices, int):
            data = tifffile.imread(self.filename, key=[indices]).squeeze()
        elif isinstance(indices, list):
            data = tifffile.imread(self.filename, key=indices).squeeze()
        else:
            indices_list = list(range(indices.start or 0, indices.stop or self.shape[0], indices.step or 1))
            data = tifffile.imread(self.filename, key=indices_list).squeeze()
        return data.astype(self.dtype)



# class MultipageTiffDataset(PMDDataset):
#     def __init__(self, filename):
#         self.filename = filename
    
#     @property
#     def shape(self):
#         return self._compute_shape(self.filename)
    
#     def _compute_shape(self, filename):
#         with tifffile.TiffFile(self.filename) as tffl:
#             num_frames = len(tffl.pages)
#             for page in tffl.pages[0:1]:
#                 image = page.asarray()
#                 x, y = page.shape
#         return (x,y,num_frames)
    
#     def get_frames(self, frames):
#         '''
#         Input: 
#             frames: a list of frames to load
#         Output: 
#             np.ndarray of dimensions (d1, d2, T) where (d1, d2) are the FOV dimensions and T is the number of frames which have been loaded
#         '''
#         return tifffile.imread(self.filename, key=frames).transpose(1, 2, 0)
        
