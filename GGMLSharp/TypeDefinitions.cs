﻿namespace GGMLSharp;

public unsafe struct ggml_init_params
{
    // memory pool
    public ulong mem_size;   // bytes
    public void* mem_buffer; // if NULL, memory will be allocated internally
    public bool no_alloc;   // don't allocate memory for the tensor data
}

public unsafe struct ggml_context
{
    public ulong mem_size;
    public void* mem_buffer;
    public bool mem_buffer_owned;
    public bool no_alloc;

    public int n_objects;

    public ggml_object* objects_begin;
    public ggml_object* objects_end;

    public ggml_scratch scratch;
    public ggml_scratch scratch_save;
}

public unsafe struct ggml_object
{
    public ulong offs;
    public ulong size;

    public ggml_object* next;

    public fixed byte padding[8];
}

public unsafe struct ggml_scratch
{
    public ulong offs;
    public ulong size;
    public void* data;
}

public unsafe struct ggml_tensor
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

    public ggml_tensor* grad;
    public ggml_tensor* src0;
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
public enum ggml_type
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
public enum ggml_op
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

public unsafe struct ggml_state
{
    const int GGML_MAX_CONTEXTS = 64;
    public Buffer64<ggml_context_container> contexts;
    //public fixed ggml_context_container contexts[GGML_MAX_CONTEXTS];
}

public struct ggml_context_container
{
    public bool used;

    public ggml_context context;
}
public unsafe static class StaticGlobal
{
    public static ggml_state* g_state = null;
    public static volatile int g_state_barrier = 0;
}

public unsafe struct block_q4_0
{
    const int QK4_0 = 32;
    public float d;          // delta
    public fixed byte qs[QK4_0 / 2];  // nibbles / quants
};
public unsafe struct block_q4_1
{
    const int QK4_1 = 32;
    public float d;          // delta
    public float m;          // min
    public fixed byte qs[QK4_1 / 2];  // nibbles / quants
};
public unsafe struct block_q4_2
{
    const int QK4_2 = 16;
    public ushort d;          // delta
    public fixed byte qs[QK4_2 / 2];  // nibbles / quants
};
public unsafe struct block_q4_3
{
    const int QK4_3 = 16;
    public ushort d;          // delta
    public ushort m;          // min
    public fixed byte qs[QK4_3 / 2];  // nibbles / quants
};
public unsafe struct block_q5_0
{
    const int QK5_0 = 32;
    public ushort d;          // delta
    public fixed byte qh[4];  // 5-th bit of quants
    public fixed byte qs[QK5_0 / 2];  // nibbles / quants
};
public unsafe struct block_q5_1
{
    const int QK5_1 = 32;
    public ushort d;          // delta
    public ushort m;          // min
    public fixed byte qh[4];  // 5-th bit of quants
    public fixed byte qs[QK5_1 / 2];  // nibbles / quants
};
public unsafe struct block_q8_0
{
    const int QK8_0 = 32;
    public float d;          // delta
    public fixed byte qs[QK8_0];  // quants
};
public unsafe struct block_q8_1
{
    const int QK8_1 = 32;
    public float d;          // delta
    public float s0;          // d * sum(qs[i]) low
    public float s1;          // d * sum(qs[i]) high
    public fixed byte qs[QK8_1];  // quants
};