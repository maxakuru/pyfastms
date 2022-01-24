cdef extern from "solver/solver.cpp":
    pass

cdef extern from "stdbool.h":
    ctypedef unsigned short bool

cdef extern from "util/image_access.h":
    cdef struct ArrayDim:
        # ArrayDim() except +
        # ArrayDim(int w, int h, int num_channels) except +
        int w
        int h
        int num_channels

        # HOST_DEVICE ArrayDim() : w(0), h(0), num_channels(0) {}
        # HOST_DEVICE ArrayDim(int w, int h, int num_channels) : w(w), h(h), num_channels(num_channels) {}
        # HOST_DEVICE Dim2D dim2d() const { return Dim2D(w, h); }
        # HOST_DEVICE size_t num_elem() const { return (size_t)w * h * num_channels; }
        # HOST_DEVICE bool operator== (const ArrayDim &other_dim) { return w == other_dim.w && h == other_dim.h && num_channels == other_dim.num_channels; }
        # HOST_DEVICE bool operator!= (const ArrayDim &other_dim) { return !(operator==(other_dim)); }

        # int w;
        # int h;
        # int num_channels;

cdef extern from "solver/solver.h":    
    cdef struct Par:
        # void fms_print "print"()

        # Length penalization parameter.
        # For bigger values:
        #   Number of discontinuities will be smaller, i.e. the solution will be smooth over larger regions.
        #   To set lambda = infinity: Set lambda to any value < 0. Solution will be a constant one, with one and the same color in every pixel ( = mean color).
        # For smaller values:
        #   More discontinuities will arise, i.e. the regions of smoothness will be smaller and the solution will resemble the input image more and more.
        #   Value lambda = 0: Solution will be equal to the original input image.
        double plambda "lambda"

        # Smoothness penalization parameter.
        # For bigger values:
        #   Solution will be more "flat" in the regions of smoothness (between the color edges), i.e. there will be only a small change in color values from pixel to pixel.
        #   To set alpha = infinity: Set alpha to any value < 0. This is the cartoon limit case, i.e. the solution will be piecewise constant. This is the segmentation case.
        # For smaller values:
        #   Solution will be more "rough" in the regions of smoothness (between the color edges), i.e. the color values are allowed to change more rapidly from pixel to pixel.
        #   Value alpha = 0: Solution will be equal to the original input image.
        double alpha

        # Temporal penalization parameter.
        # For bigger values:
        #   Solution will be driven to be similar to the previous frame solutions.
        #   To set temporal_regularization = infinity: Set to any value < 0. Solution will not change from frame to frame.
        # For smaller values:
        #   Each frame is mostly independent.
        #   Value temporal_regularization = 0: No temporal regularization, each frame is independent.
        double temporal

        # Maximal number of iterations to perform.
        # This is only an upper bound for the maximal number of iterations. The actual number of iterations may be less than this, since the iterations will be stopped once difference between consecutive solutions is small enough.
        int iterations

        # Determines the stopping criterion:
        # Iterations will be stopped if 1/(w*h) sum_{x,y,i} |u_{n+1}(x,y,i) - u_n(x,y,i)| < stopping_eps.
        double stop_eps

        # The stopping criterion will be checked for only every k-th iteration, where k is given by this parameter.
        # If set to <= 0, no checking will be performed, i.e. all max_num_iterations iterations will be made.
        int stop_k

        # If true: lambda and alpha will be adapted so that the solution will look more or less the same, for one and the same input image and for different scalings.
        #   Using this, one can run time intensive experiments on downscaled images, find suitable parameters, and then run the experiment on the original images, with the same parameters.
        #   Effectively, lambda and alpha are used as is for a "standard" image scale (640 x 480), and for a general image size w * h
        #   they are set to
        #     lambda_actual = lambda * factor
        #     alpha_actual = alpha * factor * factor
        #     where factor = sqrt(640 * 480) / sqrt(w * h).
        # If false: lambda and alpha are used as is, regardless of image scale.
        bool adapt_params

        # If true: The regularizer will be adjust to smooth less at pixels with high edge probability
        # This changes the regularizer to
        #
        #   min(alpha * |gradient u(x, y)| ^ 2, lambda * weight(x, y)),
        #
        # instead of the original regularizer with weight(x, y) = 1 for all (x, y).
        # The weight is set as
        #
        #   weight(x, y) := exp(-|gradient_of_input_image(x, y)| / sigma),
        #
        #   with  sigma := 1 / (w * h) * sum_{pixels (x, y)} |gradient_of_input_image(x, y)|.
        #
        bool weight

        # If true: The edge set will be overlaid on top of the computed color image.
        bool edges

        # Compute everything with double instead of float.
        bool use_double

        # Use CPU or CUDA.
        int engine
        # static const int engine_cpu = 0
        # static const int engine_cuda = 1

        # If true: Output information:
        #   - image dimensions
        #   - required memory
        #   - number of iterations after which the stopping criterion has been reached
        #   - time for computation only (excluding allocation and initialization)
        #   - energy
        bool verbose

    cdef cppclass SolverImplementation:
        pass

    cdef struct ArrayDim:
        pass

    cdef cppclass BaseImage:
        pass

    cdef cppclass Solver:
        Solver() except +

        BaseImage* run(BaseImage *inp, Par &par)

        void run(float *out_image, float *in_image, ArrayDim &dim, Par &par)
        void run(double *out_image, double *in_image, ArrayDim &dim, Par &par)
        # void run(unsigned char *&out_image, unsigned char *in_image, ArrayDim &dim, Par &par)