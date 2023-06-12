using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

//#undef INIT_TABLES
const int GGML_MEM_ALIGN = 16;
const int GGML_MAX_DIMS = 4;
const int GGML_MAX_NODES = 4096;
const int GGML_MAX_PARAMS = 16;
const int GGML_MAX_CONTEXTS = 64;
const int GGML_MAX_OPT = 4;
const int GGML_DEFAULT_N_THREADS = 4;
ulong GGML_OBJECT_SIZE;

const int QK4_0 = 32;
const int QK4_1 = 32;
const int QK4_2 = 16;
const int QK4_3 = 16;
const int QK5_0 = 32;
const int QK5_1 = 32;
const int QK8_0 = 32;
const int QK8_1 = 32;

// precomputed gelu table for f16 (128 KB)
ushort[] table_gelu_f16 = new ushort[1 << 16];

// precomputed silu table for f16 (128 KB)
ushort[] table_silu_f16 = new ushort[1 << 16];

// precomputed exp table for f16 (128 KB)
ushort[] table_exp_f16 = new ushort[1 << 16];

// precomputed f32 table for f16 (256 KB)
float[] table_f32_f16 = new float[1 << 16];

int[] GGML_BLCK_SIZE = new int[(int)ggml_type.GGML_TYPE_COUNT] {
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

ulong[] GGML_TYPE_SIZE = new ulong[(int)ggml_type.GGML_TYPE_COUNT] {
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

string[] GGML_TYPE_NAME = new string[(int)ggml_type.GGML_TYPE_COUNT] {
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

bool is_first_call = true;
unsafe
{
    GGML_OBJECT_SIZE = (ulong)sizeof(ggml_object);

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
Stopwatch timer;
void ggml_time_init()
{
    timer = new Stopwatch();
    timer.Start();
}
long ggml_time_ms()
{
    return timer.ElapsedMilliseconds;
}
long ggml_time_us()
{
    return timer.ElapsedTicks * 100;
}
static System.Span<TElement> InlineArrayAsSpan<TBuffer, TElement>(ref TBuffer buffer, int size) where TBuffer : struct
{
    return MemoryMarshal.CreateSpan(ref Unsafe.As<TBuffer, TElement>(ref buffer), size);
}

float GGML_COMPUTE_FP16_TO_FP32(ushort value)
{
    return (float)(System.Half)value;
}

unsafe ggml_context* ggml_init(ggml_init_params @params) {
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

            StaticGlobal.g_state = (ggml_state*)NativeMemory.AlignedAlloc((nuint)sizeof(ggml_state), GGML_MEM_ALIGN);

            // https://github.com/dotnet/csharplang/blob/main/proposals/inline-arrays.md
            ggml_context_container* basePtr = (ggml_context_container*)&StaticGlobal.g_state->contexts;
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
    ggml_context *ctx = null;

    {
        ggml_context_container* basePtr = (ggml_context_container*)&StaticGlobal.g_state->contexts;
        for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
            unsafe {
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

    ulong mem_size = (@params.mem_size + GGML_MEM_ALIGN - 1) &~(ulong)(GGML_MEM_ALIGN - 1);

    *ctx = new ggml_context {
        mem_size           = mem_size,
        mem_buffer         = @params.mem_buffer is not null ? @params.mem_buffer: NativeMemory.AlignedAlloc((nuint)mem_size, GGML_MEM_ALIGN),
        mem_buffer_owned   = @params.mem_buffer is not null ? false : true,
        no_alloc           = @params.no_alloc,
        n_objects          = 0,
        objects_begin      = null,
        objects_end        = null,
        //scratch            = { 0, 0, NULL, },
        //scratch_save       = { 0, 0, NULL, },
    };

    Debug.Assert(ctx->mem_buffer != null);

    ggml_assert_aligned(ctx->mem_buffer);

    GGML_PRINT_DEBUG($"{nameof(ggml_init)}: context initialized\n");

    ggml_critical_section_end();

    return ctx;
}

unsafe void ggml_free(ggml_context *ctx) {
    // make this function thread safe
    ggml_critical_section_start();

    bool found = false;
    ggml_context_container* basePtr = (ggml_context_container*)&StaticGlobal.g_state->contexts;
    for (int i = 0; i < GGML_MAX_CONTEXTS; i++)
    unsafe {
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

static void ggml_critical_section_start()
{
    int processing = Interlocked.Exchange(ref StaticGlobal.g_state_barrier, StaticGlobal.g_state_barrier + 1);

    while (processing > 0)
    {
        // wait for other threads to finish
        Interlocked.Decrement(ref StaticGlobal.g_state_barrier);
        //sched_yield(); // TODO: reconsider this
        Thread.Yield();
        processing = Interlocked.Exchange(ref StaticGlobal.g_state_barrier, StaticGlobal.g_state_barrier + 1);
    }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
static void ggml_critical_section_end()
{
    Interlocked.Decrement(ref StaticGlobal.g_state_barrier);
}

void GGML_PRINT_DEBUG(string format, params object?[]? args)
{
    Console.Write(format, args);
}

void GGML_PRINT(string format, params object?[]? args)
{
    Console.Write(format, args);
}

unsafe void ggml_print_object(in ggml_object *obj) {
    GGML_PRINT($" - ggml_object: offset = {obj->offs}, size = {obj->size}, next = {(nint)obj->next}\n");
}

unsafe void ggml_print_objects(in ggml_context *ctx) {
    ggml_object *obj = ctx->objects_begin;

    GGML_PRINT($"{nameof(ggml_print_objects)}: objects in context {(nint)ctx}:\n");

    while (obj != null)
    {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_PRINT($"{nameof(ggml_print_objects)}: --- end ---\n");
}

unsafe void ggml_assert_aligned(void* ptr)
{
    Debug.Assert(((nint)ptr) % GGML_MEM_ALIGN == 0);
}

////////////////////////////////////////////////////////////////////////////////

unsafe ggml_tensor *ggml_new_tensor_impl(
        ggml_context *ctx,
        ggml_type type,
        int    n_dims,
        in long* ne,
        void*  data) {
    // always insert objects at the end of the context's memory pool
    ggml_object *obj_cur = ctx->objects_end;

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
ggml_object* obj_new = (ggml_object *)(mem_buffer + cur_end);

if (ctx->scratch.data == null || data != null)
{
    size_needed += (ulong)sizeof(ggml_tensor);

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size)
    {
        GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + size_needed + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
        Debug.Assert(false);
        return null;
    }

    *obj_new = new ggml_object {
        offs = cur_end + GGML_OBJECT_SIZE,
        size = size_needed,
        next = null,
    };
    } else {
        if (ctx->scratch.offs + size_needed > ctx->scratch.size)
        {
            GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the scratch memory\n");
            Debug.Assert(false);
            return null;
        }

        if (cur_end + (ulong)sizeof(ggml_tensor) +GGML_OBJECT_SIZE > ctx->mem_size) {
            GGML_PRINT($"{nameof(ggml_new_tensor_impl)}: not enough space in the context's memory pool (needed {cur_end + (ulong)sizeof(ggml_tensor) + GGML_OBJECT_SIZE}, available {ctx->mem_size})\n");
            Debug.Assert(false);
            return null;
        }

        data = (byte*) ctx->scratch.data + ctx->scratch.offs;

        *obj_new = new ggml_object {
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

    ggml_tensor * result = (ggml_tensor *)(mem_buffer + obj_new->offs);

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

unsafe ggml_tensor * ggml_new_tensor(
        ggml_context * ctx,
          ggml_type type,
        int    n_dims,
        long* ne) {
    return ggml_new_tensor_impl(ctx, type, n_dims, ne, null);
}

unsafe ggml_tensor *ggml_new_tensor_1d(
        ggml_context *ctx,
        ggml_type type,
        long ne0) {
    return ggml_new_tensor(ctx, type, 1, &ne0);
}

unsafe ggml_tensor *ggml_new_tensor_2d(
        ggml_context *ctx,
        ggml_type type,
        long ne0,
        long ne1) {
    long* ne = stackalloc long[2] { ne0, ne1 };
    return ggml_new_tensor(ctx, type, 2, ne);
}

unsafe ggml_tensor* ggml_new_tensor_3d(
        ggml_context *ctx,
        ggml_type type,
        long ne0,
        long ne1,
        long ne2)
{
    long* ne = stackalloc long[3] { ne0, ne1, ne2 };
    return ggml_new_tensor(ctx, type, 3, ne);
}

unsafe ggml_tensor *ggml_new_tensor_4d(
        ggml_context * ctx,
        ggml_type type,
        long ne0,
        long ne1,
        long ne2,
        long ne3) {
    long* ne = stackalloc long[4] { ne0, ne1, ne2, ne3 };
    return ggml_new_tensor(ctx, type, 4, ne);
}

unsafe struct ggml_init_params
{
    // memory pool
    public ulong mem_size;   // bytes
    public void* mem_buffer; // if NULL, memory will be allocated internally
    public bool no_alloc;   // don't allocate memory for the tensor data
}

unsafe struct ggml_context
{
    public ulong mem_size;
    public void* mem_buffer;
    public bool mem_buffer_owned;
    public bool no_alloc;

    public int n_objects;

    public ggml_object * objects_begin;
    public ggml_object * objects_end;

    public ggml_scratch scratch;
    public ggml_scratch scratch_save;
}

unsafe struct ggml_object
{
    public ulong offs;
    public ulong size;

    public ggml_object* next;

    public fixed byte padding[8];
}

unsafe struct ggml_scratch
{
    public ulong offs;
    public ulong size;
    public void* data;
}

unsafe struct ggml_tensor
{
    const int GGML_MAX_DIMS = 4;
    const int GGML_MAX_OPT = 4;
    public ggml_type type;

    public int n_dims;
    public fixed long ne[GGML_MAX_DIMS]; // number of elements
    public fixed ulong nb[GGML_MAX_DIMS]; // stride in bytes:
                                         // nb[0] = sizeof(type)
                                         // nb[1] = nb[0]   * ne[0] + padding
                                         // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    public ggml_op op;

    public bool is_param;

    public ggml_tensor * grad;
    public ggml_tensor * src0;
    public ggml_tensor* src1;
    //public fixed ggml_tensor* opt[GGML_MAX_OPT];
    public fixed long opt[GGML_MAX_OPT];

    // thread scheduling
    public int n_tasks;

    // performance
    public int perf_runs;
    public long perf_cycles;
    public long perf_time_us;

    public void* data;
    public fixed byte padding[8];
}
enum ggml_type
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4,
    GGML_TYPE_Q4_3 = 5,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
}

// available tensor operations:
enum ggml_op
{
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_SUM,
    GGML_OP_MEAN,
    GGML_OP_REPEAT,
    GGML_OP_ABS,
    GGML_OP_SGN,
    GGML_OP_NEG,
    GGML_OP_STEP,
    GGML_OP_RELU,
    GGML_OP_GELU,
    GGML_OP_SILU,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,

    GGML_OP_MUL_MAT,

    GGML_OP_SCALE,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_SOFT_MAX,
    GGML_OP_ROPE,
    GGML_OP_ALIBI,
    GGML_OP_CONV_1D_1S,
    GGML_OP_CONV_1D_2S,

    GGML_OP_FLASH_ATTN,
    GGML_OP_FLASH_FF,

    GGML_OP_MAP_UNARY,
    GGML_OP_MAP_BINARY,

    GGML_OP_COUNT,
}

[System.Runtime.CompilerServices.InlineArray(64 /*GGML_MAX_CONTEXTS*/)]
public struct Buffer64<T>
{
    private T _element0;
}


unsafe struct ggml_state
{
    const int GGML_MAX_CONTEXTS = 64;
    public Buffer64<ggml_context_container> contexts;
    //public fixed ggml_context_container contexts[GGML_MAX_CONTEXTS];
}

struct ggml_context_container
{
    public bool used;

    public ggml_context context;
}
unsafe static class StaticGlobal
{
    public static ggml_state* g_state = null;
    public static volatile int g_state_barrier = 0;
}
unsafe struct block_q4_0
{
    const int QK4_0 = 32;
    public float d;          // delta
    public fixed byte qs[QK4_0 / 2];  // nibbles / quants
};
unsafe struct block_q4_1
{
    const int QK4_1 = 32;
    public float d;          // delta
    public float m;          // min
    public fixed byte qs[QK4_1 / 2];  // nibbles / quants
};
unsafe struct block_q4_2
{
    const int QK4_2 = 16;
    public ushort d;          // delta
    public fixed byte qs[QK4_2 / 2];  // nibbles / quants
};
unsafe struct block_q4_3
{
    const int QK4_3 = 16;
    public ushort d;          // delta
    public ushort m;          // min
    public fixed byte qs[QK4_3 / 2];  // nibbles / quants
};
unsafe struct block_q5_0
{
    const int QK5_0 = 32;
    public ushort d;          // delta
    public fixed byte qh[4];  // 5-th bit of quants
    public fixed byte qs[QK5_0 / 2];  // nibbles / quants
};
unsafe struct block_q5_1
{
    const int QK5_1 = 32;
    public ushort d;          // delta
    public ushort m;          // min
    public fixed byte qh[4];  // 5-th bit of quants
    public fixed byte qs[QK5_1 / 2];  // nibbles / quants
};
unsafe struct block_q8_0
{
    const int QK8_0 = 32;
    public float d;          // delta
    public fixed byte qs[QK8_0];  // quants
};
unsafe struct block_q8_1
{
    const int QK8_1 = 32;
    public float d;          // delta
    public float s0;          // d * sum(qs[i]) low
    public float s1;          // d * sum(qs[i]) high
    public fixed byte qs[QK8_1];  // quants
};