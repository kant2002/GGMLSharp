using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace GGMLSharp;

public static unsafe class Ggml
{
    private static ggml_state* g_state = null;
    private static volatile int g_state_barrier = 0;
    private static Stopwatch timer;
    private static bool is_first_call = true;

    const int GGML_MEM_ALIGN = 16;
    const int GGML_MAX_DIMS = 4;
    const int GGML_MAX_NODES = 4096;
    const int GGML_MAX_PARAMS = 16;
    const int GGML_MAX_CONTEXTS = 64;
    const int GGML_MAX_OPT = 4;
    const int GGML_DEFAULT_N_THREADS = 4;

    const int QK4_0 = 32;
    const int QK4_1 = 32;
    const int QK4_2 = 16;
    const int QK4_3 = 16;
    const int QK5_0 = 32;
    const int QK5_1 = 32;
    const int QK8_0 = 32;
    const int QK8_1 = 32;
    private static ulong GGML_OBJECT_SIZE = (ulong)sizeof(ggml_object);

    // precomputed gelu table for f16 (128 KB)
    private static ushort[] table_gelu_f16 = new ushort[1 << 16];

    // precomputed silu table for f16 (128 KB)
    private static ushort[] table_silu_f16 = new ushort[1 << 16];

    // precomputed exp table for f16 (128 KB)
    private static ushort[] table_exp_f16 = new ushort[1 << 16];

    // precomputed f32 table for f16 (256 KB)
    private static float[] table_f32_f16 = new float[1 << 16];

    private static int[] GGML_BLCK_SIZE = new int[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_F32]  = */1,
        /*[GGML_TYPE_F16]  = */1,
        /*[GGML_TYPE_Q4_0] = */QK4_0,
        /*[GGML_TYPE_Q4_1] = */QK4_1,
        /*[GGML_TYPE_Q4_2] = */QK4_2,
        /*[GGML_TYPE_Q4_3] = */QK4_3,
        /*[GGML_TYPE_Q5_0] = */QK5_0,
        /*[GGML_TYPE_Q5_1] = */QK5_1,
        /*[GGML_TYPE_Q8_0] = */QK8_0,
        /*[GGML_TYPE_Q8_1] = */QK8_1,
        /*[GGML_TYPE_I8]   = */1,
        /*[GGML_TYPE_I16]  = */1,
        /*[GGML_TYPE_I32]  = */1,
    };

    private static ulong[] GGML_TYPE_SIZE = new ulong[(int)ggml_type.GGML_TYPE_COUNT] {
        /* [GGML_TYPE_F32]  = */ sizeof(float),
        /* [GGML_TYPE_F16]  = */ sizeof(short)                                  /*sizeof(ggml_fp16_t)*/,
        /* [GGML_TYPE_Q4_0] = */ sizeof(float) + QK4_0 / 2                      /*sizeof(block_q4_0) */,
        /* [GGML_TYPE_Q4_1] = */ 2 * sizeof(float) + QK4_1 / 2                  /*sizeof(block_q4_1)*/,
        /* [GGML_TYPE_Q4_2] = */ sizeof(short) + QK4_2 / 2                      /*sizeof(block_q4_2)*/,
        /* [GGML_TYPE_Q4_3] = */ 2 * sizeof(short) + QK4_3 / 2                  /*sizeof(block_q4_3)*/,
        /* [GGML_TYPE_Q5_0] = */ sizeof(short) + sizeof(uint) + QK5_0 / 2       /*sizeof(block_q5_0)*/,
        /* [GGML_TYPE_Q5_1] = */ 2 * sizeof(short) + sizeof(uint) + QK5_1 / 2   /*sizeof(block_q5_1)*/,
        /* [GGML_TYPE_Q8_0] = */ sizeof(float) + QK8_0                          /*sizeof(block_q8_0)*/,
        /* [GGML_TYPE_Q8_1] = */ 3*sizeof(float) + QK8_1                        /*sizeof(block_q8_1)*/,
        /* [GGML_TYPE_I8]   = */ sizeof(byte),
        /* [GGML_TYPE_I16]  = */ sizeof(short),
        /* [GGML_TYPE_I32]  = */ sizeof(int),
    };

    private static string[] GGML_TYPE_NAME = new string[(int)ggml_type.GGML_TYPE_COUNT] {
        /*[GGML_TYPE_F32]  = */ "f32",
        /*[GGML_TYPE_F16]  = */ "f16",
        /*[GGML_TYPE_Q4_0] = */ "q4_0",
        /*[GGML_TYPE_Q4_1] = */ "q4_1",
        /*[GGML_TYPE_Q4_2] = */ "q4_2",
        /*[GGML_TYPE_Q4_3] = */ "q4_3",
        /*[GGML_TYPE_Q5_0] = */ "q5_0",
        /*[GGML_TYPE_Q5_1] = */ "q5_1",
        /*[GGML_TYPE_Q8_0] = */ "q8_0",
        /*[GGML_TYPE_Q8_1] = */ "q8_1",
        /*[GGML_TYPE_I8]   = */ "i8",
        /*[GGML_TYPE_I16]  = */ "i16",
        /*[GGML_TYPE_I32]  = */ "i32",
    };

    private static string[] GGML_OP_SYMBOL = new string[(int)ggml_op.GGML_OP_COUNT] {
        "none",

        "x",
        "x+y",
        "x-y",
        "x*y",
        "x/y",
        "x^2",
        "√x",
        "Σx",
        "Σx/n",
        "repeat(x)",
        "abs(x)",
        "sgn(x)",
        "-x",
        "step(x)",
        "relu(x)",
        "gelu(x)",
        "silu(x)",
        "norm(x)",
        "rms_norm(x)",

        "X*Y",

        "x*v",
        "x-\\>y",
        "cont(x)",
        "reshape(x)",
        "view(x)",
        "permute(x)",
        "transpose(x)",
        "get_rows(x)",
        "diag_mask_inf(x)",
        "soft_max(x)",
        "rope(x)",
        "alibi(x)",
        "conv_1d_1s(x)",
        "conv_1d_2s(x)",

        "flash_attn(x)",
        "flash_ff(x)",

        "f(x)",
        "f(x,y)",
    };

    public static ggml_context* ggml_init(ggml_init_params @params)
    {
        // make this function thread safe
        ggml_critical_section_start();

        if (is_first_call)
        {
            // initialize time system (required on Windows)
            ggml_time_init();

            // initialize GELU, SILU and EXP F32 tables
            {
                long t_start = ggml_time_us(); //UNUSED(t_start);

                ushort ii;
                for (int i = 0; i < (1 << 16); ++i)
                {
                    ushort ui = (ushort)i;
                    //memcpy(&ii, &ui, sizeof(ushort));
                    ii = ui; // Probably different configurations define differently, that's why C code use memcpy.
                    float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
#if INIT_TABLES
                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
                table_exp_f16[i] = GGML_FP32_TO_FP16(expf(f));
#endif
                }

                long t_end = ggml_time_us();

                GGML_PRINT_DEBUG($"{nameof(ggml_init)}: GELU, SILU and EXP tables initialized in {(t_end - t_start) / 1000.0f} ms\n");
            }

            // initialize g_state
            unsafe
            {
                long t_start = ggml_time_us();

                g_state = (ggml_state*)NativeMemory.AlignedAlloc((nuint)sizeof(ggml_state), GGML_MEM_ALIGN);

                // https://github.com/dotnet/csharplang/blob/main/proposals/inline-arrays.md
                ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
                for (int i = 0; i < GGML_MAX_CONTEXTS; ++i)
                    unsafe
                    {
                        var itemPtr = basePtr + i;
                        itemPtr->used = false;
                    }

                long t_end = ggml_time_us();
                GGML_PRINT_DEBUG($"{nameof(ggml_init)}: g_state initialized in {(t_end - t_start) / 1000.0f} ms\n");
            }

            // initialize cuBLAS
#if GGML_USE_CUBLAS
        ggml_init_cublas();
#elif GGML_USE_CLBLAST
        ggml_cl_init();
#endif

            is_first_call = false;
        }

        // find non-used context in g_state
        ggml_context* ctx = null;

        {
            ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
            for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
                unsafe
                {
                    var itemPtr = basePtr + i;
                    if (!itemPtr->used)
                    {
                        itemPtr->used = true;
                        ctx = &itemPtr->context;

                        GGML_PRINT_DEBUG($"{nameof(ggml_init)}: found unused context {i}\n");
                        break;
                    }
                }
        }

        if (ctx == null)
        {
            GGML_PRINT_DEBUG($"{nameof(ggml_init)}: no unused context found\n");

            ggml_critical_section_end();

            return null;
        }

        ulong mem_size = (@params.mem_size + GGML_MEM_ALIGN - 1) & ~(ulong)(GGML_MEM_ALIGN - 1);

        *ctx = new ggml_context
        {
            mem_size = mem_size,
            mem_buffer = @params.mem_buffer is not null ? @params.mem_buffer : NativeMemory.AlignedAlloc((nuint)mem_size, GGML_MEM_ALIGN),
            mem_buffer_owned = @params.mem_buffer is not null ? false : true,
            no_alloc = @params.no_alloc,
            n_objects = 0,
            objects_begin = null,
            objects_end = null,
            //scratch            = { 0, 0, NULL, },
            //scratch_save       = { 0, 0, NULL, },
        };

        Debug.Assert(ctx->mem_buffer != null);

        ggml_assert_aligned(ctx->mem_buffer);

        GGML_PRINT_DEBUG($"{nameof(ggml_init)}: context initialized\n");

        ggml_critical_section_end();

        return ctx;
    }

    public static void ggml_free(ggml_context* ctx)
    {
        // make this function thread safe
        ggml_critical_section_start();

        bool found = false;
        ggml_context_container* basePtr = (ggml_context_container*)&g_state->contexts;
        for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
            unsafe
            {
                var itemPtr = basePtr + i;
                if (&itemPtr->context == ctx)
                {
                    itemPtr->used = false;

                    GGML_PRINT_DEBUG($"{nameof(ggml_free)}: context {i} with {ctx->n_objects} objects has been freed. memory used = {ctx->objects_end->offs + ctx->objects_end->size}\n");

                    if (ctx->mem_buffer_owned)
                    {
                        //GGML_ALIGNED_FREE(ctx->mem_buffer);
                        NativeMemory.AlignedFree(ctx->mem_buffer);
                    }

                    found = true;
                    break;
                }
            }

        if (!found)
        {
            GGML_PRINT_DEBUG($"{nameof(ggml_free)}: context not found\n");
        }

        ggml_critical_section_end();
    }

    public static void ggml_print_object(in ggml_object* obj)
    {
        GGML_PRINT($" - ggml_object: offset = {obj->offs}, size = {obj->size}, next = {(nint)obj->next}\n");
    }

    public static void ggml_print_objects(in ggml_context* ctx)
    {
        ggml_object* obj = ctx->objects_begin;

        GGML_PRINT($"{nameof(ggml_print_objects)}: objects in context {(nint)ctx}:\n");

        while (obj != null)
        {
            ggml_print_object(obj);
            obj = obj->next;
        }

        GGML_PRINT($"{nameof(ggml_print_objects)}: --- end ---\n");
    }

    public static ggml_tensor* ggml_new_tensor(
            ggml_context* ctx,
              ggml_type type,
            int n_dims,
            long* ne)
    {
        return ggml_new_tensor_impl(ctx, type, n_dims, ne, null);
    }

    public static ggml_tensor* ggml_new_tensor_1d(
            ggml_context* ctx,
            ggml_type type,
            long ne0)
    {
        return ggml_new_tensor(ctx, type, 1, &ne0);
    }

    public static ggml_tensor* ggml_new_tensor_2d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1)
    {
        long* ne = stackalloc long[2] { ne0, ne1 };
        return ggml_new_tensor(ctx, type, 2, ne);
    }

    public static ggml_tensor* ggml_new_tensor_3d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1,
            long ne2)
    {
        long* ne = stackalloc long[3] { ne0, ne1, ne2 };
        return ggml_new_tensor(ctx, type, 3, ne);
    }

    public static ggml_tensor* ggml_new_tensor_4d(
            ggml_context* ctx,
            ggml_type type,
            long ne0,
            long ne1,
            long ne2,
            long ne3)
    {
        long* ne = stackalloc long[4] { ne0, ne1, ne2, ne3 };
        return ggml_new_tensor(ctx, type, 4, ne);
    }

    public static ggml_tensor* ggml_new_i32(ggml_context* ctx, int value)
    {
        ctx->scratch_save = ctx->scratch;
        ctx->scratch.data = null;

        ggml_tensor* result = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I32, 1);

        ctx->scratch = ctx->scratch_save;

        ggml_set_i32(result, value);

        return result;
    }

    public static ggml_tensor* ggml_new_f32(ggml_context* ctx, float value)
    {
        ctx->scratch_save = ctx->scratch;
        ctx->scratch.data = null;

        ggml_tensor* result = ggml_new_tensor_1d(ctx, ggml_type.GGML_TYPE_I32, 1);

        ctx->scratch = ctx->scratch_save;

        ggml_set_f32(result, value);

        return result;
    }

    public static ggml_tensor* ggml_dup_tensor(ggml_context* ctx, in ggml_tensor* src)
    {
        return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, null);
    }

    public static ggml_tensor* ggml_set_zero(ggml_tensor* tensor)
    {
        NativeMemory.Fill(tensor->data, ggml_nbytes(tensor), 0);
        return tensor;
    }

    public static ggml_tensor* ggml_set_i32(ggml_tensor* tensor, int value)
    {
        nuint n = (nuint)ggml_nrows(tensor);
        int nc = (int)tensor->ne[0];
        nuint n1 = (nuint)tensor->nb[1];

        byte* data = (byte*)tensor->data;

        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i8(nc, (byte*)(data + i * n1), (byte)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i16(nc, (short*)(data + i * n1), (short)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i32(nc, (int*)(data + i * n1), value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f16(nc, (Half*)(data + i * n1), value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
                    }
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return tensor;
    }

    public static ggml_tensor* ggml_set_f32(ggml_tensor* tensor, float value)
    {
        nuint n = (nuint)ggml_nrows(tensor);
        int nc = (int)tensor->ne[0];
        nuint n1 = (nuint)tensor->nb[1];

        byte* data = (byte*)tensor->data;

        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i8(nc, (byte*)(data + i * n1), (byte)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i16(nc, (short*)(data + i * n1), (short)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_i32(nc, (int*)(data + i * n1), (int)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f16(nc, (Half*)(data + i * n1), (int)value);
                    }
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    for (nuint i = 0; i < n; i++)
                    {
                        ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
                    }
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return tensor;
    }

    public static void ggml_vec_set_i8(int n, byte* x, byte v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_i16(int n, short* x, short v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_i32(int n, int* x, int v) { for (int i = 0; i < n; ++i) x[i] = v; }

    public static void ggml_vec_set_f16(int n, Half* x, int v) { for (int i = 0; i < n; ++i) x[i] = (Half)v; }
    public static void ggml_vec_set_f32(int n, float* x, float v) { for (int i = 0; i < n; ++i) x[i] = v; }


    public static float ggml_get_f32_1d(ggml_tensor* tensor, int i)
    {
        switch (tensor->type)
        {
            case ggml_type.GGML_TYPE_I8:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(byte));
                    return ((byte*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_I16:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(short));
                    return ((short*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_I32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(int));
                    return ((int*)(tensor->data))[i];
                }
                break;
            case ggml_type.GGML_TYPE_F16:
                {
                    Debug.Assert(tensor->nb[0] == (ulong)sizeof(Half));
                    return (float)(((Half*)(tensor->data))[i]);
                }
                break;
            case ggml_type.GGML_TYPE_F32:
                {
                    Debug.Assert(tensor->nb[0] == sizeof(float));
                    return ((float*)(tensor->data))[i];
                }
                break;
            default:
                {
                    Debug.Assert(false);
                }
                break;
        }

        return 0.0f;
    }

    // check if node is part of the graph
    private static bool ggml_graph_find(ggml_cgraph* cgraph, ggml_tensor* node) {
        if (cgraph == null) {
            return true;
        }

        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (cgraph->get_node(i) == node) {
                return true;
            }
        }

        return false;
    }

    private static ggml_tensor* ggml_graph_get_parent(ggml_cgraph* cgraph, ggml_tensor* node) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor* parent = cgraph->get_node(i);

            if (parent->grad == node) {
                return parent;
            }
        }

        return null;
    }

    public static void ggml_graph_dump_dot(ggml_cgraph* gb, ggml_cgraph* gf, string filename) {
        string color;

        using var fp = new StreamWriter(File.OpenWrite(filename));

        fp.WriteLine("digraph G {");
        fp.WriteLine("  newrank = true;");
        fp.WriteLine("  rankdir = LR;");

        for (int i = 0; i < gb->n_nodes; i++) {
            ggml_tensor* node = gb->get_node(i);

            if (ggml_graph_get_parent(gb, node) != null) {
                continue;
            }

            if (node->is_param) {
                color = "yellow";
            } else if (node->grad is not null) {
                if (ggml_graph_find(gf, node))
                {
                    color = "green";
                } else
                {
                    color = "lightblue";
                }
            } else
            {
                color = "white";
            }

            fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"{i} [{node->ne[0]}, {node->ne[1]} | <x>{GGML_OP_SYMBOL[(int)node->op]}");

            if (node->grad is not null) {
                fp.WriteLine($" | <g>{GGML_OP_SYMBOL[(int)node->grad->op]}\"; ]");
            } else {
                fp.WriteLine("\"; ]");
            }
        }

        for (int i = 0; i < gb->n_leafs; i++) {
            ggml_tensor* node = gb->get_leaf(i);

            color = "pink";

            if (ggml_nelements(node) == 1) {
                fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>{(double)ggml_get_f32_1d(node, 0)}\"; ]");
            } else {
                fp.WriteLine($"  \"{(nint)node}\" [ style = filled; fillcolor = {color}; shape = record; label=\"<x>CONST {i} [{node->ne[0]}, {node->ne[1]}]\"; ]");
            }
        }

        for (int i = 0; i < gb->n_nodes; i++) {
            ggml_tensor* node = gb->get_node(i);

            ggml_tensor* parent = ggml_graph_get_parent(gb, node);

            if (node->src0 is not null) {
                ggml_tensor* parent0 = ggml_graph_get_parent(gb, node->src0);

                fp.WriteLine("  \"{0}\":{1} -> \"{2}\":{3} [ arrowhead = {4}; style = {5}; label = \"x\"; ]",
                    parent0 is not null ? (nint)parent0 : (nint)node->src0,
                    parent0 is not null ? "g" : "x",
                    parent is not null ? (nint)parent : (nint)node,
                    parent is not null ? "g" : "x",
                    parent is not null ? "empty" : "vee",
                    parent is not null ? "dashed" : "solid");
            }

            if (node->src1 is not null) {
                ggml_tensor* parent1 = ggml_graph_get_parent(gb, node->src1);

                fp.WriteLine("  \"{0}\":{1} -> \"{2}\":{3} [ arrowhead = {4}; style = {5}; label = \"x\"; ]",
                        parent1 is not null ? (nint)parent1 : (nint)node->src1,
                        parent1 is not null ? "g" : "x",
                        parent is not null ? (nint)parent : (nint)node,
                        parent is not null ? "g" : "x",
                        parent is not null ? "empty" : "vee",
                        parent is not null ? "dashed" : "solid");
            }
        }

        for (int i = 0; i < gb->n_leafs; i++) {
            ggml_tensor* node = gb->get_leaf(i);

            if (node->src0 is not null) {
                fp.WriteLine("  \"{0}\":x -> \"{1}\":x [ label = \"x\"; ]\n",
                        (nint)node->src0,
                        (nint)node);
            }

            if (node->src1 is not null) {
                fp.WriteLine("  \"{0}\":x -> \"{1}\":x [ label = \"y\"; ]\n",
                        (nint)node->src1,
                        (nint)node);
            }
        }

        fp.WriteLine("}\n");

        GGML_PRINT($"{nameof(ggml_graph_dump_dot)}: dot -Tpng {filename} -o {filename}.png && open {filename}.png\n");
    }

    public static void ggml_graph_reset(ggml_cgraph* cgraph)
    {
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            ggml_tensor* grad = cgraph->get_grad(i);

            if (grad is not null)
            {
                ggml_set_zero(grad);
            }
        }
    }

    public static ggml_tensor* ggml_view_tensor(
        ggml_context* ctx,
        ggml_tensor* src)
    {
        ggml_tensor* result = ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);

        result->nb[0] = src->nb[0];
        result->nb[1] = src->nb[1];
        result->nb[2] = src->nb[2];
        result->nb[3] = src->nb[3];

        return result;
    }


    public static long ggml_nelements(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
        return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    }

    public static int ggml_nrows(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return (int)(tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
    }

    public static nuint ggml_nbytes(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return ((nuint)ggml_nelements(tensor) * (nuint)GGML_TYPE_SIZE[(int)tensor->type]) / (nuint)GGML_BLCK_SIZE[(int)tensor->type];
    }

    public static void ggml_set_param(
        ggml_context* ctx,
        ggml_tensor* tensor)
    {
        tensor->is_param = true;

        Debug.Assert(tensor->grad == null);
        tensor->grad = ggml_dup_tensor(ctx, tensor);
    }

    public static ggml_tensor* ggml_add(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_add_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_add_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_add_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_sub(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_sub_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_sub_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_sub_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_mul(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_mul_inplace(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_div(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_div_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_div_inplace(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_div_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_sqr(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sqr_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sqr_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sqr_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sqrt(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sqrt_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sqrt_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sqrt_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sum(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            is_node = true;
        }

        ggml_tensor* result = ggml_new_tensor_1d(ctx, a->type, 1);

        result->op = ggml_op.GGML_OP_SUM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    public static ggml_tensor* ggml_mean(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement
            is_node = true;
        }

        long* ne = stackalloc long[GGML_MAX_DIMS] { 1, a->ne[1], a->ne[2], a->ne[3] };
        ggml_tensor* result = ggml_new_tensor(ctx, ggml_type.GGML_TYPE_F32, a->n_dims, ne);

        result->op = ggml_op.GGML_OP_MEAN;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    public static ggml_tensor* ggml_repeat(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        Debug.Assert(ggml_can_repeat(a, b));

        bool is_node = false;

        if (a->grad is not null)
        {
            is_node = true;
        }

        if (ggml_are_same_shape(a, b) && !is_node)
        {
            return a;
        }

        ggml_tensor* result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

        result->op = ggml_op.GGML_OP_REPEAT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    public static ggml_tensor* ggml_abs(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_abs_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_abs_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_abs_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_sgn(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_sgn_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_sgn_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_sgn_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_neg(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_neg_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_neg_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_neg_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_step(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_step_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_step_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_step_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_relu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_relu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_relu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_relu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_gelu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_gelu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_gelu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_gelu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_silu(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_silu_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_silu_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_silu_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_norm(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_norm_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_norm_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_norm_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_rms_norm(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_rms_norm_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_rms_norm_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_rms_norm_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_mul_mat(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_mul_mat_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_mul_mat_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_mul_mat_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_scale(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_scale_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_scale_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_scale_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_cpy(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b)
    {
        return ggml_cpy_impl(ctx, a, b, false);
    }

    public static ggml_tensor* ggml_cpy_inplace(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b)
    {
        return ggml_cpy_impl(ctx, a, b, true);
    }

    public static ggml_tensor* ggml_cont(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        return ggml_cont_impl(ctx, a, false);
    }

    public static ggml_tensor* ggml_cont_inplace(
            ggml_context* ctx,
            ggml_tensor* a)
    {
        return ggml_cont_impl(ctx, a, true);
    }

    public static ggml_tensor* ggml_transpose(
        ggml_context* ctx,
        ggml_tensor* a)
    {
        bool is_node = false;

        if (a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = ggml_view_tensor(ctx, a);

        result->ne[0] = a->ne[1];
        result->ne[1] = a->ne[0];

        result->nb[0] = a->nb[1];
        result->nb[1] = a->nb[0];

        result->op = ggml_op.GGML_OP_TRANSPOSE;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static void ggml_compute_backward(ggml_context* ctx, ggml_tensor* tensor, bool inplace)
    {
        ggml_tensor* src0 = tensor->src0;
        ggml_tensor* src1 = tensor->src1;

        switch (tensor->op)
        {
            case ggml_op.GGML_OP_DUP:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_ADD:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad = ggml_add_impl(ctx, src1->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SUB:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad = ggml_sub_impl(ctx, src1->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_MUL:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx, src1, tensor->grad),
                                    inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_add_impl(ctx,
                                    src1->grad,
                                    ggml_mul(ctx, src0, tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_DIV:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_div(ctx, tensor->grad, src1),
                                    inplace);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_sub_impl(ctx,
                                    src1->grad,
                                    ggml_mul(ctx,
                                        tensor->grad,
                                        ggml_div(ctx, tensor, src1)),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SQR:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx,
                                        ggml_mul(ctx, src0, tensor->grad),
                                        ggml_repeat(ctx, ggml_new_f32(ctx, 2.0f), src0)),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SQRT:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_div(ctx,
                                        ggml_repeat(ctx, ggml_new_f32(ctx, 0.5f), tensor),
                                        tensor),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SUM:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_repeat(ctx, tensor->grad, src0->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_MEAN:
                {
                    Debug.Assert(false); // TODO: implement
                }
                break;
            case ggml_op.GGML_OP_REPEAT:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_sum(ctx, tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_ABS:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad =
                            ggml_add_impl(ctx,
                                    src0->grad,
                                    ggml_mul(ctx,
                                        ggml_sgn(ctx, src0),
                                        tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SGN:
                {
                    if (src0->grad is not null)
                    {
                        // noop
                    }
                }
                break;
            case ggml_op.GGML_OP_NEG:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_sub_impl(ctx, src0->grad, tensor->grad, inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_STEP:
                {
                    if (src0->grad is not null)
                    {
                        // noop
                    }
                }
                break;
            case ggml_op.GGML_OP_RELU:
                {
                    if (src0->grad is not null)
                    {
                        src0->grad = ggml_sub_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_step(ctx, src0),
                                    tensor->grad),
                                inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_GELU:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_ALIBI:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_SILU:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_NORM:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_RMS_NORM:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_MUL_MAT:
                {
                    if (src0->grad is not null)
                    {
                        // TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor->grad);
                        Debug.Assert(false);
                    }
                    if (src1->grad is not null)
                    {
                        src1->grad =
                            ggml_add_impl(ctx,
                                    src1->grad,
                                    ggml_mul_mat(ctx,
                                        ggml_cont(ctx, ggml_transpose(ctx, src0)),
                                        tensor->grad),
                                    inplace);
                    }
                }
                break;
            case ggml_op.GGML_OP_SCALE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CPY:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONT:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_RESHAPE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_VIEW:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_PERMUTE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_TRANSPOSE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_GET_ROWS:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_DIAG_MASK_INF:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_SOFT_MAX:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_ROPE:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONV_1D_1S:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_CONV_1D_2S:
                {
                    Debug.Assert(false); // TODO: not implemented
                }
                break;
            case ggml_op.GGML_OP_FLASH_ATTN:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_FLASH_FF:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_MAP_UNARY:
            case ggml_op.GGML_OP_MAP_BINARY:
                {
                    Debug.Assert(false); // not supported
                }
                break;
            case ggml_op.GGML_OP_NONE:
                {
                    // nop
                }
                break;
            case ggml_op.GGML_OP_COUNT:
                {
                    Debug.Assert(false);
                }
                break;
        }
    }

    private static void ggml_visit_parents(ggml_cgraph* cgraph, ggml_tensor* node)
    {
        if (node->grad == null)
        {
            // this usually happens when we generate intermediate nodes from constants in the backward pass
            // it can also happen during forward pass, if the user performs computations with constants
            if (node->op != ggml_op.GGML_OP_NONE)
            {
                //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
            }
        }

        // check if already visited
        for (int i = 0; i < cgraph->n_nodes; i++)
        {
            if (cgraph->get_node(i) == node)
            {
                return;
            }
        }

        for (int i = 0; i < cgraph->n_leafs; i++)
        {
            if (cgraph->get_leaf(i) == node)
            {
                return;
            }
        }

        if (node->src0 is not null)
        {
            ggml_visit_parents(cgraph, node->src0);
        }

        if (node->src1 is not null)
        {
            ggml_visit_parents(cgraph, node->src1);
        }

        for (int i = 0; i < GGML_MAX_OPT; ++i)
        {
            if (node->opt[i] != 0)
            {
                ggml_visit_parents(cgraph, (ggml_tensor*)node->opt[i]);
            }
        }

        if (node->op == ggml_op.GGML_OP_NONE && node->grad == null)
        {
            // reached a leaf node, not part of the gradient graph (e.g. a constant)
            Debug.Assert(cgraph->n_leafs < GGML_MAX_NODES);

            cgraph->set_leaf(cgraph->n_leafs, node);
            cgraph->n_leafs++;
        }
        else
        {
            Debug.Assert(cgraph->n_nodes < GGML_MAX_NODES);

            cgraph->set_node(cgraph->n_nodes, node);
            cgraph->set_grad(cgraph->n_nodes, node->grad);
            cgraph->n_nodes++;
        }
    }

    private static void ggml_build_forward_impl(ggml_cgraph* cgraph, ggml_tensor* tensor, bool expand)
    {
        if (!expand)
        {
            cgraph->n_nodes = 0;
            cgraph->n_leafs = 0;
        }

        int n0 = cgraph->n_nodes;

        ggml_visit_parents(cgraph, tensor);

        int n_new = cgraph->n_nodes - n0;
        GGML_PRINT_DEBUG($"{nameof(ggml_build_forward_impl)}: visited {n_new} new nodes\n");

        if (n_new > 0)
        {
            // the last added node should always be starting point
            Debug.Assert(cgraph->get_node(cgraph->n_nodes - 1) == tensor);
        }
    }

    public static void ggml_build_forward_expand(ggml_cgraph* cgraph, ggml_tensor* tensor)
    {
        ggml_build_forward_impl(cgraph, tensor, true);
    }

    public static ggml_cgraph ggml_build_forward(ggml_tensor* tensor)
    {
        ggml_cgraph result = new ggml_cgraph()
        {
            n_nodes = 0,
            n_leafs = 0,
            n_threads = GGML_DEFAULT_N_THREADS,
            work_size = 0,
            work = null,
            // nodes        = { NULL },
            // grads        = { NULL },
            // leafs        = { NULL },
            perf_runs = 0,
            perf_cycles = 0,
            perf_time_us = 0,
        };

        ggml_build_forward_impl(&result, tensor, false);

        return result;
    }

    public static ggml_cgraph ggml_build_backward(ggml_context* ctx, ggml_cgraph* gf, bool keep)
    {
        ggml_cgraph result = *gf;

        Debug.Assert(gf->n_nodes > 0);

        // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
        if (keep)
        {
            for (int i = 0; i < gf->n_nodes; i++)
            {
                ggml_tensor* node = gf->get_node(i);

                if (node->grad is not null)
                {
                    node->grad = ggml_dup_tensor(ctx, node);
                    gf->set_grad(i, node->grad);
                }
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = gf->get_node(i);

            // because we detached the grad nodes from the original graph, we can afford inplace operations
            if (node->grad is not null)
            {
                ggml_compute_backward(ctx, node, keep);
            }
        }

        for (int i = gf->n_nodes - 1; i >= 0; i--)
        {
            ggml_tensor* node = gf->get_node(i);

            if (node->is_param)
            {
                GGML_PRINT_DEBUG($"{nameof(ggml_build_backward)}: found root node {(nuint)node}\n");
                ggml_build_forward_impl(&result, node->grad, true);
            }
        }

        return result;
    }
    ////////////////////////////////////////////////////////////////////////////////

    static unsafe ggml_tensor* ggml_new_tensor_impl(
            ggml_context* ctx,
            ggml_type type,
            int n_dims,
            in long* ne,
            void* data)
    {
        // always insert objects at the end of the context's memory pool
        ggml_object* obj_cur = ctx->objects_end;

        ulong cur_offs = obj_cur == null ? 0 : obj_cur->offs;
        ulong cur_size = obj_cur == null ? 0 : obj_cur->size;
        ulong cur_end = cur_offs + cur_size;

        ulong size_needed = 0;

        if (data == null && !ctx->no_alloc)
        {
            size_needed += GGML_TYPE_SIZE[(int)type] * (ulong)(ne[0] / GGML_BLCK_SIZE[(int)type]);
            for (int i = 1; i < n_dims; i++)
            {
                size_needed *= (ulong)ne[i];
            }
            // align to GGML_MEM_ALIGN
            size_needed = ((size_needed + GGML_MEM_ALIGN - 1) / GGML_MEM_ALIGN) * GGML_MEM_ALIGN;
        }

        byte* mem_buffer = (byte*)ctx->mem_buffer;
        ggml_object* obj_new = (ggml_object*)(mem_buffer + cur_end);

        if (ctx->scratch.data == null || data != null)
        {
            size_needed += (ulong)sizeof(ggml_tensor);

            if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + size_needed + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
                Debug.Assert(false);
                return null;
            }

            *obj_new = new ggml_object
            {
                offs = cur_end + GGML_OBJECT_SIZE,
                size = size_needed,
                next = null,
            };
        }
        else
        {
            if (ctx->scratch.offs + size_needed > ctx->scratch.size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the scratch memory\n");
                Debug.Assert(false);
                return null;
            }

            if (cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE > ctx->mem_size)
            {
                GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
                Debug.Assert(false);
                return null;
            }

            data = (byte*)ctx->scratch.data + ctx->scratch.offs;

            *obj_new = new ggml_object
            {
                offs = cur_end + GGML_OBJECT_SIZE,
                size = (ulong)sizeof(ggml_tensor),
                next = null,
            };

            //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

            ctx->scratch.offs += size_needed;
        }

        if (obj_cur != null)
        {
            obj_cur->next = obj_new;
        }
        else
        {
            // this is the first object in this context
            ctx->objects_begin = obj_new;
        }

        ctx->objects_end = obj_new;

        //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

        ggml_tensor* result = (ggml_tensor*)(mem_buffer + obj_new->offs);

        ggml_assert_aligned(result);

        ggml_tensor res = new ggml_tensor
        {
            /*.type         =*/
            type = type,
            /*.n_dims       =*/
            n_dims = n_dims,
            //ne = { 1, 1, 1, 1 },
            //nb = { 0, 0, 0, 0 },
            op = ggml_op.GGML_OP_NONE,
            is_param = false,
            grad = null,
            src0 = null,
            src1 = null,
            //opt = { NULL },
            n_tasks = 0,
            perf_runs = 0,
            perf_cycles = 0,
            perf_time_us = 0,
            data = (data == null && !ctx->no_alloc) ? (void*)(result + 1) : data,
            //pad = { 0 },
        };
        res.ne[0] = 1;
        res.ne[1] = 1;
        res.ne[2] = 1;
        res.ne[3] = 1;
        *result = res;

        // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
        //ggml_assert_aligned(result->data);

        for (int i = 0; i < n_dims; i++)
        {
            result->ne[i] = ne[i];
        }

        result->nb[0] = GGML_TYPE_SIZE[(int)type];
        result->nb[1] = result->nb[0] * (ulong)(result->ne[0] / GGML_BLCK_SIZE[(int)type]);
        for (int i = 2; i < GGML_MAX_DIMS; i++)
        {
            result->nb[i] = result->nb[i - 1] * (ulong)result->ne[i - 1];
        }

        ctx->n_objects++;

        return result;
    }

    private static ggml_tensor* ggml_add_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_ADD;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_sub_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SUB;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_mul_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        if (inplace)
        {
            Debug.Assert(is_node == false);
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_MUL;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_div_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_are_same_shape(a, b));

        bool is_node = false;

        if (!inplace && (a->grad is not null || b->grad is not null))
        {
            is_node = true;
        }

        if (inplace)
        {
            Debug.Assert(is_node == false);
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_DIV;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_sqr_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SQR;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_sqrt_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SQRT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_abs_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_ABS;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_sgn_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SGN;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_neg_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_NEG;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_step_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_STEP;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_relu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_RELU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_gelu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_GELU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_silu_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SILU;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    private static ggml_tensor* ggml_norm_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_NORM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null; // TODO: maybe store epsilon here?

        return result;
    }

    private static ggml_tensor* ggml_rms_norm_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && (a->grad is not null))
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_RMS_NORM;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null; // TODO: maybe store epsilon here?

        return result;
    }

    private static ggml_tensor* ggml_mul_mat_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_can_mul_mat(a, b));
        Debug.Assert(!ggml_is_transposed(a));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            is_node = true;
        }

        long* ne = stackalloc long[4] { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
        ggml_tensor* result = ggml_new_tensor(ctx, ggml_type.GGML_TYPE_F32, Math.Min(a->n_dims, b->n_dims), ne);

        result->op = ggml_op.GGML_OP_MUL_MAT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_scale_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_is_scalar(b));
        Debug.Assert(ggml_is_padded_1d(a));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            is_node = true;
        }

        // TODO: when implement backward, fix this:
        //ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
        ggml_tensor* result = ggml_view_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_SCALE;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_cpy_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            ggml_tensor* b,
            bool inplace)
    {
        Debug.Assert(ggml_nelements(a) == ggml_nelements(b));
        bool is_node = false;

        if (a->grad is not null || b->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        // make a view of the destination
        ggml_tensor* result = ggml_view_tensor(ctx, b);

        result->op = ggml_op.GGML_OP_CPY;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = b;

        return result;
    }

    private static ggml_tensor* ggml_cont_impl(
            ggml_context* ctx,
            ggml_tensor* a,
            bool inplace)
    {
        bool is_node = false;

        if (!inplace && a->grad is not null)
        {
            Debug.Assert(false); // TODO: implement backward
            is_node = true;
        }

        ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

        result->op = ggml_op.GGML_OP_CONT;
        result->grad = is_node ? ggml_dup_tensor(ctx, result) : null;
        result->src0 = a;
        result->src1 = null;

        return result;
    }

    static bool ggml_is_scalar(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_is_vector(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_is_matrix(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return tensor->ne[2] == 1 && tensor->ne[3] == 1;
    }

    static bool ggml_can_mul_mat(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
    }

    static bool ggml_is_transposed(ggml_tensor* tensor)
    {
        return tensor->nb[0] > tensor->nb[1];
    }

    static bool ggml_is_contiguous(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            tensor->nb[0] == GGML_TYPE_SIZE[(int)tensor->type] &&
            tensor->nb[1] == (tensor->nb[0] * (ulong)tensor->ne[0]) / (ulong)GGML_BLCK_SIZE[(int)tensor->type] &&
            tensor->nb[2] == tensor->nb[1] * (ulong)tensor->ne[1] &&
            tensor->nb[3] == tensor->nb[2] * (ulong)tensor->ne[2];
    }

    static bool ggml_is_padded_1d(ggml_tensor* tensor)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            tensor->nb[0] == GGML_TYPE_SIZE[(int)tensor->type] &&
            tensor->nb[2] == tensor->nb[1] * (ulong)tensor->ne[1] &&
            tensor->nb[3] == tensor->nb[2] * (ulong)tensor->ne[2];
    }

    private static bool ggml_are_same_shape(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
    }

    // check if t1 can be represented as a repeatition of t0
    private static bool ggml_can_repeat(ggml_tensor* t0, ggml_tensor* t1)
    {
        Debug.Assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

        return
            (t1->ne[0] % t0->ne[0] == 0) &&
            (t1->ne[1] % t0->ne[1] == 0) &&
            (t1->ne[2] % t0->ne[2] == 0) &&
            (t1->ne[3] % t0->ne[3] == 0);
    }

    static void ggml_assert_aligned(void* ptr)
    {
        Debug.Assert(((nint)ptr) % GGML_MEM_ALIGN == 0);
    }

    static void ggml_time_init()
    {
        timer = new Stopwatch();
        timer.Start();
    }
    static long ggml_time_ms()
    {
        return timer.ElapsedMilliseconds;
    }
    static long ggml_time_us()
    {
        return timer.ElapsedTicks * 100;
    }
    static System.Span<TElement> InlineArrayAsSpan<TBuffer, TElement>(ref TBuffer buffer, int size) where TBuffer : struct
    {
        return MemoryMarshal.CreateSpan(ref Unsafe.As<TBuffer, TElement>(ref buffer), size);
    }

    static float GGML_COMPUTE_FP16_TO_FP32(ushort value)
    {
        return (float)(System.Half)value;
    }

    static void ggml_critical_section_start()
    {
        int processing = Interlocked.Exchange(ref g_state_barrier, g_state_barrier + 1);

        while (processing > 0)
        {
            // wait for other threads to finish
            Interlocked.Decrement(ref g_state_barrier);
            //sched_yield(); // TODO: reconsider this
            Thread.Yield();
            processing = Interlocked.Exchange(ref g_state_barrier, g_state_barrier + 1);
        }
    }

    // TODO: make this somehow automatically executed
    //       some sort of "sentry" mechanism
    static void ggml_critical_section_end()
    {
        Interlocked.Decrement(ref g_state_barrier);
    }

    static void GGML_PRINT_DEBUG(string format, params object?[]? args)
    {
        Console.Write(format, args);
    }

    static void GGML_PRINT(string format, params object?[]? args)
    {
        Console.Write(format, args);
    }
}
