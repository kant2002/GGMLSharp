using GGMLSharp;
using System.Diagnostics;
using static GGMLSharp.Ggml;

unsafe
{
    ggml_init_params init_params = default;
    {
        init_params.mem_size = 128 * 1024 * 1024;
        init_params.mem_buffer = null;
        init_params.no_alloc = false;
    };

    ggml_context* ctx0 = ggml_init(init_params);

    {
        ggml_tensor *x = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, 1);

        ggml_set_param(ctx0, x);

        ggml_tensor *a = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, 1);
        ggml_tensor *b = ggml_mul(ctx0, x, x);
        ggml_tensor *f = ggml_mul(ctx0, b, a);

        // a*x^2
        // 2*a*x

        ggml_print_objects(ctx0);

        ggml_cgraph gf = ggml_build_forward(f);
        ggml_cgraph gb = ggml_build_backward(ctx0, &gf, false);

        ggml_set_f32(x, 2.0f);
        ggml_set_f32(a, 3.0f);

        ggml_graph_reset(&gf);
        ggml_set_f32(f->grad, 1.0f);

        ggml_graph_compute(ctx0, &gb);

        Console.WriteLine("f     = {0:F}", ggml_get_f32_1d(f, 0));
        Console.WriteLine("df/dx = {0:F}", ggml_get_f32_1d(x->grad, 0));

        Debug.Assert(ggml_get_f32_1d(f, 0) == 12.0f);
        Debug.Assert(ggml_get_f32_1d(x->grad, 0) == 12.0f);

        ggml_set_f32(x, 3.0f);

        ggml_graph_reset(&gf);
        ggml_set_f32(f->grad, 1.0f);

        ggml_graph_compute(ctx0, &gb);

        Console.WriteLine("f     = {0:F}", ggml_get_f32_1d(f, 0));
        Console.WriteLine("df/dx = {0:F}", ggml_get_f32_1d(x->grad, 0));

        Debug.Assert(ggml_get_f32_1d(f, 0) == 27.0f);
        Debug.Assert(ggml_get_f32_1d(x->grad, 0) == 18.0f);

        ggml_graph_dump_dot(&gf, null, "test1-1-forward.dot");
        ggml_graph_dump_dot(&gb, &gf, "test1-1-backward.dot");
    }

    return 0;
}