using GGMLSharp;
using System.Diagnostics;
using static GGMLSharp.Ggml;

ulong next = 1;
int RAND_MAX = 32767;

unsafe
{
    ggml_init_params init_params = default;
    {
        init_params.mem_size = 1024 * 1024 * 1024;
        init_params.mem_buffer = null;
        init_params.no_alloc = false;
    };

    ggml_opt_params opt_params = ggml_opt_default_params(ggml_opt_type.GGML_OPT_LBFGS);
    //ggml_opt_params opt_params = ggml_opt_default_params(ggml_opt_type.GGML_OPT_ADAM);

    opt_params.n_threads = (args.Length > 0) ? int.Parse(args[1]) : 8;

    const int NP = 1 << 12;
    const int NF = 1 << 8;

    ggml_context * ctx0 = ggml_init(init_params);

    ggml_tensor * F = ggml_new_tensor_2d(ctx0, ggml_type.GGML_TYPE_F32, NF, NP);
    ggml_tensor * l = ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, NP);

    // regularization weight
    ggml_tensor * lambda = ggml_new_f32(ctx0, 1e-5f);

    xsrand(0);

    for (int j = 0; j < NP; j++) {
        float ll = j < NP/2 ? 1.0f : -1.0f;
        ((float *)l->data)[j] = ll;

        for (int i = 0; i < NF; i++) {
            ((float *)F->data)[j*NF + i] = ((ll > 0 && i < NF/2 ? 1.0f : ll < 0 && i >= NF/2 ? 1.0f : 0.0f) + ((float)xrand()/(float)RAND_MAX - 0.5f)*0.1f)/(0.5f*NF);
        }
    }

    {
        // initial guess
        ggml_tensor * x = ggml_set_f32(ggml_new_tensor_1d(ctx0, ggml_type.GGML_TYPE_F32, NF), 0.0f);

        ggml_set_param(ctx0, x);

        // f = sum_j[(f_j*x - l)^2]/n + lambda*|x^2|
        ggml_tensor* f =
            ggml_add(ctx0,
                ggml_div(ctx0,
                    ggml_sum(ctx0,
                        ggml_sqr(ctx0,
                            ggml_sub(ctx0,
                                ggml_mul_mat(ctx0, F, x),
                                l)
                            )
                        ),
                    ggml_new_f32(ctx0, NP)
                    ),
                ggml_mul(ctx0,
                    ggml_sum(ctx0, ggml_sqr(ctx0, x)),
                    lambda)
                );

        ggml_opt_result res = ggml_opt(null, opt_params, f);

        Debug.Assert(res == ggml_opt_result.GGML_OPT_OK);

        // print results
        for (int i = 0; i < 16; i++) {
            Console.WriteLine($"x[{i,3}] = {0:F6}", ((float *)x->data)[i]);
        }
        Console.WriteLine("...");
        for (int i = NF - 16; i < NF; i++) {
            Console.WriteLine($"x[{i,3}] = {0:F6}", ((float *)x->data)[i]);
        }
        Console.WriteLine();

        for (int i = 0; i < NF; ++i) {
            if (i < NF/2) {
                Debug.Assert(is_close(((float *)x->data)[i],  1.0f, 1e-2f));
            } else {
                Debug.Assert(is_close(((float *)x->data)[i], -1.0f, 1e-2f));
            }
        }
    }

    ggml_free(ctx0);
}

static bool is_close(float a, float b, float epsilon) {
    return Math.Abs(a - b) < epsilon;
}

int xrand() // RAND_MAX assumed to be 32767
{
    next = next * 214013L + 2531011;
    return (int)((next >> 16) & 0x7FFF);
}

void xsrand(uint seed)
{
    next = seed;
}