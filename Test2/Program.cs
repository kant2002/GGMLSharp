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
    
    //ggml_opt_params opt_params = ggml_opt_default_params(ggml_opt_type.GGML_OPT_LBFGS);

    ggml_opt_params opt_params = ggml_opt_default_params(ggml_opt_type.GGML_OPT_ADAM);
    opt_params.adam.alpha = 0.01f;

    // original threads: 8
    int nthreads = 8;
    string env = Environment.GetEnvironmentVariable("GGML_NTHREADS");
    if (!string.IsNullOrWhiteSpace(env)) {
        nthreads = int.Parse(env);
    }
    if (args.Length > 1) {
        nthreads = int.Parse(args[0]);
    }
    opt_params.n_threads = nthreads;
    Console.WriteLine($"test2: n_threads:{opt_params.n_threads}");

    float[] xi = new []{  1.0f,  2.0f,  3.0f,  4.0f,  5.0f , 6.0f,  7.0f,  8.0f,  9.0f,  10.0f, };
    float[] yi = new []{ 15.0f, 25.0f, 35.0f, 45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, };

    int n = xi.Length;

    ggml_context * ctx0 = ggml_init(init_params);

    ggml_tensor * x = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, n);
    ggml_tensor * y = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, n);

    for (int i = 0; i < n; i++) {
        ((float *) x->data)[i] = xi[i];
        ((float *) y->data)[i] = yi[i];
    }
    
    {
        ggml_tensor * t0 = ggml_new_f32(ctx0, 0.0f);
        ggml_tensor * t1 = ggml_new_f32(ctx0, 0.0f);

        // initialize auto-diff parameters:
        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = sum_i[(t0 + t1*x_i - y_i)^2]/(2n)
        ggml_tensor * f =
            ggml_div(ctx0,
                ggml_sum(ctx0,
                    ggml_sqr(ctx0,
                        ggml_sub(ctx0,
                            ggml_add(ctx0,
                                ggml_mul(ctx0, x, ggml_repeat(ctx0, t1, x)),
                                ggml_repeat(ctx0, t0, x)),
                            y)
                    )
                ),
                ggml_new_f32(ctx0, 2.0f*n));

        ggml_opt_result res = ggml_opt(null, opt_params, f);

        Debug.Assert(res == ggml_opt_result.GGML_OPT_OK);

        Console.WriteLine("t0 = {0:F6}", ggml_get_f32_1d(t0, 0));
        Console.WriteLine("t1 = {0:F6}", ggml_get_f32_1d(t1, 0));

        Debug.Assert(is_close(ggml_get_f32_1d(t0, 0),  5.0f, 1e-3f));
        Debug.Assert(is_close(ggml_get_f32_1d(t1, 0), 10.0f, 1e-3f));
    }

    {
        ggml_tensor * t0 = ggml_new_f32(ctx0, -1.0f);
        ggml_tensor * t1 = ggml_new_f32(ctx0,  9.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = 0.5*sum_i[abs(t0 + t1*x_i - y_i)]/n
        ggml_tensor * f =
            ggml_mul(ctx0,
                    ggml_new_f32(ctx0, 1.0f/(2*n)),
                    ggml_sum(ctx0,
                        ggml_abs(ctx0,
                            ggml_sub(ctx0,
                                ggml_add(ctx0,
                                    ggml_mul(ctx0, x, ggml_repeat(ctx0, t1, x)),
                                    ggml_repeat(ctx0, t0, x)),
                                y)
                            )
                        )
                    );


        ggml_opt_result res = ggml_opt(null, opt_params, f);

        Debug.Assert(res == ggml_opt_result.GGML_OPT_OK);
        Debug.Assert(is_close(ggml_get_f32_1d(t0, 0),  5.0f, 1e-2f));
        Debug.Assert(is_close(ggml_get_f32_1d(t1, 0), 10.0f, 1e-2f));
    }

    {
        ggml_tensor * t0 = ggml_new_f32(ctx0,  5.0f);
        ggml_tensor * t1 = ggml_new_f32(ctx0, -4.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = t0^2 + t1^2
        ggml_tensor * f =
            ggml_add(ctx0,
                    ggml_sqr(ctx0, t0),
                    ggml_sqr(ctx0, t1)
                    );

        ggml_opt_result res = ggml_opt(null, opt_params, f);

        Debug.Assert(res == ggml_opt_result.GGML_OPT_OK);
        Debug.Assert(is_close(ggml_get_f32_1d(f,  0), 0.0f, 1e-3f));
        Debug.Assert(is_close(ggml_get_f32_1d(t0, 0), 0.0f, 1e-3f));
        Debug.Assert(is_close(ggml_get_f32_1d(t1, 0), 0.0f, 1e-3f));
    }

    /////////////////////////////////////////

    {
        ggml_tensor * t0 = ggml_new_f32(ctx0, -7.0f);
        ggml_tensor * t1 = ggml_new_f32(ctx0,  8.0f);

        ggml_set_param(ctx0, t0);
        ggml_set_param(ctx0, t1);

        // f = (t0 + 2*t1 - 7)^2 + (2*t0 + t1 - 5)^2
        ggml_tensor * f =
            ggml_add(ctx0,
                    ggml_sqr(ctx0,
                        ggml_sub(ctx0,
                            ggml_add(ctx0,
                                t0,
                                ggml_mul(ctx0, t1, ggml_new_f32(ctx0, 2.0f))),
                            ggml_new_f32(ctx0, 7.0f)
                            )
                        ),
                    ggml_sqr(ctx0,
                        ggml_sub(ctx0,
                            ggml_add(ctx0,
                                ggml_mul(ctx0, t0, ggml_new_f32(ctx0, 2.0f)),
                                t1),
                            ggml_new_f32(ctx0, 5.0f)
                            )
                        )
                    );

        ggml_opt_result res = ggml_opt(null, opt_params, f);

        Debug.Assert(res == ggml_opt_result.GGML_OPT_OK);
        Debug.Assert(is_close(ggml_get_f32_1d(f,  0), 0.0f, 1e-3f));
        Debug.Assert(is_close(ggml_get_f32_1d(t0, 0), 1.0f, 1e-3f));
        Debug.Assert(is_close(ggml_get_f32_1d(t1, 0), 3.0f, 1e-3f));
    }

    ggml_free(ctx0);
}

static bool is_close(float a, float b, float epsilon) {
    return Math.Abs(a - b) < epsilon;
}