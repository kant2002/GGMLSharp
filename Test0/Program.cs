using GGMLSharp;
using System.Diagnostics;
using static GGMLSharp.Ggml;

//#undef INIT_TABLES

unsafe
{
    ggml_init_params init_params = default;
    {
        init_params.mem_size = 128 * 1024 * 1024;
        init_params.mem_buffer = null;
        init_params.no_alloc = false;
    };

    ggml_context* ctx0 = ggml_init(init_params);

    ggml_tensor* t1 = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, 10);
    ggml_tensor* t2 = ggml_new_tensor_2d(ctx0, ggml_type.GGML_TYPE_I16, 10, 20);
    ggml_tensor* t3 = ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_I32, 10, 20, 30);

    Debug.Assert(t1->n_dims == 1);
    Debug.Assert(t1->ne[0] == 10);
    Debug.Assert(t1->nb[1] == 10 * sizeof(float));

    Debug.Assert(t2->n_dims == 2);
    Debug.Assert(t2->ne[0] == 10);
    Debug.Assert(t2->ne[1] == 20);
    Debug.Assert(t2->nb[1] == 10 * sizeof(Int16));
    Debug.Assert(t2->nb[2] == 10 * 20 * sizeof(Int16));

    Debug.Assert(t3->n_dims == 3);
    Debug.Assert(t3->ne[0] == 10);
    Debug.Assert(t3->ne[1] == 20);
    Debug.Assert(t3->ne[2] == 30);
    Debug.Assert(t3->nb[1] == 10 * sizeof(Int32));
    Debug.Assert(t3->nb[2] == 10 * 20 * sizeof(Int32));
    Debug.Assert(t3->nb[3] == 10 * 20 * 30 * sizeof(Int32));

    ggml_print_objects(ctx0);

    ggml_free(ctx0);

    return 0;
}