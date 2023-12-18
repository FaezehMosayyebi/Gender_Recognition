
class Iterator(PyDataset):
    """Base class for image data iterators.

    DEPRECATED.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    Args:
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(
                "Asked to retrieve element {idx}, "
                "but the Sequence "
                "has length {length}".format(idx=idx, length=len(self))
            )
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[
                current_index : current_index + self.batch_size
            ]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        raise NotImplementedError


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension.

    Args:
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    Yields:
        Tuple of (root, filename) with extension in `white_list_formats`.
    """

    def _recursive_list(subpath):
        return sorted(
            os.walk(subpath, followlinks=follow_links), key=lambda x: x[0]
        )

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith(".tiff"):
                warnings.warn(
                    'Using ".tiff" files with multiple bands '
                    "will cause distortion. Please verify your output."
                )
            if fname.lower().endswith(white_list_formats):
                yield root, fname


def _list_valid_filenames_in_directory(
    directory, white_list_formats, split, class_indices, follow_links
):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    Args:
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.

    Returns:
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        all_files = list(
            _iter_valid_files(directory, white_list_formats, follow_links)
        )
        num_files = len(all_files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = all_files[start:stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links
        )
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)

    return classes, filenames

class BatchFromFilesMixin:
    """Adds methods related to getting batches from filenames.

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(
        self,
        image_data_generator,
        target_size,
        color_mode,
        data_format,
        save_to_dir,
        save_prefix,
        save_format,
        subset,
        interpolation,
        keep_aspect_ratio,
    ):
        """Sets attributes to use later for processing files into a batch.

        Args:
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images
            to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic". If
                PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        if color_mode not in {"rgb", "rgba", "grayscale"}:
            raise ValueError(
                f"Invalid color mode: {color_mode}"
                '; expected "rgb", "rgba", or "grayscale".'
            )
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == "rgba":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == "rgb":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == "validation":
                split = (0, validation_split)
            elif subset == "training":
                split = (validation_split, 1)
            else:
                raise ValueError(
                    f"Invalid subset name: {subset};"
                    'expected "training" or "validation"'
                )
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape, dtype=self.dtype
        )
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = image_utils.load_img(
                filepaths[j],
                color_mode=self.color_mode,
                target_size=self.target_size,
                interpolation=self.interpolation,
                keep_aspect_ratio=self.keep_aspect_ratio,
            )
            x = image_utils.img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, "close"):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = image_utils.array_to_img(
                    batch_x[i], self.data_format, scale=True
                )
                fname = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == "input":
            batch_y = batch_x.copy()
        elif self.class_mode in {"binary", "sparse"}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == "categorical":
            batch_y = np.zeros(
                (len(batch_x), len(self.class_indices)), dtype=self.dtype
            )
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.0
        elif self.class_mode == "multi_output":
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == "raw":
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    @property
    def filepaths(self):
        """List of absolute paths to image files."""
        raise NotImplementedError(
            "`filepaths` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation."""
        raise NotImplementedError(
            "`labels` property method has not been implemented in {}.".format(
                type(self).__name__
            )
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            "`sample_weight` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )


class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.

    DEPRECATED.
    """

    allowed_class_modes = {"categorical", "binary", "sparse", "input", None}

    def __init__(
        self,
        directory,
        image_data_generator,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
        )
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError(
                "Invalid class_mode: {}; expected one of: {}".format(
                    class_mode, self.allowed_class_modes
                )
            )
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(
                    _list_valid_filenames_in_directory,
                    (
                        dirpath,
                        self.white_list_formats,
                        self.split,
                        self.class_indices,
                        follow_links,
                    ),
                )
            )
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype="int32")
        for classes in classes_list:
            self.classes[i : i + len(classes)] = classes
            i += len(classes)

        io_utils.print_msg(
            f"Found {self.samples} images belonging to "
            f"{self.num_classes} classes."
        )
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super().__init__(self.samples, batch_size, shuffle, seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


class FaceMasker:
    """DEPRECATED."""

    def __init__(
        self,
        low_threshold,
        high_threshold,
        kernel_size,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size
    
    def flow_from_directory(
        self,
        directory,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
    ):
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            keep_aspect_ratio=keep_aspect_ratio,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dtype=self.dtype,
        )

    def flow_from_dataframe(
        self,
        dataframe,
        directory=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
        **kwargs,
    ):
        if "has_ext" in kwargs:
            warnings.warn(
                "has_ext is deprecated, filenames in the dataframe have "
                "to match the exact filenames in disk.",
                DeprecationWarning,
            )
        if "sort" in kwargs:
            warnings.warn(
                "sort is deprecated, batches will be created in the"
                "same order than the filenames provided if shuffle"
                "is set to False.",
                DeprecationWarning,
            )
        if class_mode == "other":
            warnings.warn(
                '`class_mode` "other" is deprecated, please use '
                '`class_mode` "raw".',
                DeprecationWarning,
            )
            class_mode = "raw"
        if "drop_duplicates" in kwargs:
            warnings.warn(
                "drop_duplicates is deprecated, you can drop duplicates "
                "by using the pandas.DataFrame.drop_duplicates method.",
                DeprecationWarning,
            )

        return DataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            dtype=self.dtype,
        )

    
    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        Args:
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        Returns:
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(
            x,
            transform_parameters.get("theta", 0),
            transform_parameters.get("tx", 0),
            transform_parameters.get("ty", 0),
            transform_parameters.get("shear", 0),
            transform_parameters.get("zx", 1),
            transform_parameters.get("zy", 1),
            row_axis=img_row_axis,
            col_axis=img_col_axis,
            channel_axis=img_channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval,
            order=self.interpolation_order,
        )

        if transform_parameters.get("channel_shift_intensity") is not None:
            x = apply_channel_shift(
                x,
                transform_parameters["channel_shift_intensity"],
                img_channel_axis,
            )

        if transform_parameters.get("flip_horizontal", False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get("flip_vertical", False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get("brightness") is not None:
            x = apply_brightness_shift(
                x, transform_parameters["brightness"], False
            )

        return x

    
    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits the data generator to some sample data.

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.

        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.

        When `rescale` is set to a value, rescaling is applied to
        sample data before computing the internal data stats.

        Args:
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError(
                "Input to `.fit()` should have rank 4. Got array with shape: "
                + str(x.shape)
            )
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                "Expected input to be images (as Numpy array) "
                'following the data format convention "'
                + self.data_format
                + '" (channels on axis '
                + str(self.channel_axis)
                + "), i.e. expected either 1, 3 or 4 channels on axis "
                + str(self.channel_axis)
                + ". However, it was passed an array with shape "
                + str(x.shape)
                + " ("
                + str(x.shape[self.channel_axis])
                + " channels)."
            )

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if self.rescale:
            x *= self.rescale

        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype,
            )
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= self.std + 1e-6

        if self.zca_whitening:
            n = len(x)
            flat_x = np.reshape(x, (n, -1))

            u, s, _ = np.linalg.svd(flat_x.T, full_matrices=False)
            s_inv = np.sqrt(n) / (s + self.zca_epsilon)
            self.zca_whitening_matrix = (u * s_inv).dot(u.T)



def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.

    DEPRECATED.

    Args:
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    Returns:
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)




def apply_affine_transform(
    x,
    theta=0,
    tx=0,
    ty=0,
    shear=0,
    zx=1,
    zy=1,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    order=1,
):
    """Applies an affine transformation specified by the parameters given.

    DEPRECATED.
    """
    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError(
            "'row_axis', 'col_axis', and 'channel_axis' must be distinct"
        )

    # shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(
            f"Invalid axis' indices: {actual_indices - valid_indices}"
        )

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError(
            "Channels are allowed and the first and last dimensions."
        )

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array(
            [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]]
        )
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w
        )
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [
            scipy.ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval,
            )
            for x_channel in x
        ]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x