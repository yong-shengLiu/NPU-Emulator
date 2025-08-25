typedef struct packed {
    ara_op_e op; // Operation

    // Stores and slides do not re-shuffle the
    // source registers. In these two cases, vl refers
    // to the target EEW and vtype.vsew, respectively.
    // Since operand requesters work with the old
    // eew of the source registers, we should rescale
    // vl to the old eew to fetch the correct number of Bytes.
    //
    // Another solution would be to pass directly the target
    // eew (vstores) or the vtype.vsew (vslides), but this would
    // create confusion with the current naming convention
    logic scale_vl;

    // Mask vector register operand
    logic vm;
    rvv_pkg::vew_e eew_vmask;

    // 1st vector register operand
    logic [4:0] vs1;
    logic use_vs1;
    opqueue_conversion_e conversion_vs1;
    rvv_pkg::vew_e eew_vs1;
    rvv_pkg::vew_e old_eew_vs1;

    // 2nd vector register operand
    logic [4:0] vs2;
    logic use_vs2;
    opqueue_conversion_e conversion_vs2;
    rvv_pkg::vew_e eew_vs2;

    // Use vd as an operand as well (e.g., vmacc)
    logic use_vd_op;
    rvv_pkg::vew_e eew_vd_op;

    // Scalar operand
    elen_t scalar_op;
    logic use_scalar_op;

    // 2nd scalar operand: stride for constant-strided vector load/stores, slide offset for vector
    // slides
    elen_t stride;
    logic is_stride_np2;

    // Destination vector register
    logic [4:0] vd;
    logic use_vd;

    // If asserted: vs2 is kept in MulFPU opqueue C, and vd_op in MulFPU A
    logic swap_vs2_vd_op;

    // Effective length multiplier
    rvv_pkg::vlmul_e emul;

    // Number of segments in segment mem op
    logic [2:0] nf;

    // Is this a fault-only-first load?
    logic fault_only_first;

    // Rounding-Mode for FP operations
    fpnew_pkg::roundmode_e fp_rm;
    // Widen FP immediate (re-encoding)
    logic wide_fp_imm;
    // Resizing of FP conversions
    resize_e cvt_resize;

    // Vector machine metadata
    vlen_t vl;
    vlen_t vstart;
    rvv_pkg::vtype_t vtype;

    // Request token, for registration in the sequencer
    logic token;
  } ara_req_t;