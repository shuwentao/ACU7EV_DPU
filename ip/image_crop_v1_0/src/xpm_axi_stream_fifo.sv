// Local wrapper mapping Xilinx XPM macro name "xpm_axi_stream_fifo" (used by
// image_crop.sv) onto the real XPM primitive "xpm_fifo_axis" defined in
// Vivado's $VIVADO/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv.
//
// This file is ONLY needed for 3rd-party simulators (VCS/Questa/...) where the
// XPM macro wrapper is not auto-generated. Vivado's own flow does not need it.
`timescale 1ns / 1ps

module xpm_axi_stream_fifo #(
  parameter integer TDATA_WIDTH      = 32,
  parameter integer PROG_FULL_THRESH = 10,
  parameter integer FIFO_DEPTH       = 2048,
  parameter         PACKET_FIFO      = "false",
  parameter         FIFO_MEMORY_TYPE = "auto"
) (
  input  wire                    s_aclk,
  input  wire                    s_aresetn,
  input  wire                    m_aclk,
  input  wire                    s_axis_tvalid,
  output wire                    s_axis_tready,
  input  wire [TDATA_WIDTH-1:0]  s_axis_tdata,
  input  wire                    s_axis_tlast,
  input  wire                    s_axis_tuser,
  output wire                    m_axis_tvalid,
  input  wire                    m_axis_tready,
  output wire [TDATA_WIDTH-1:0]  m_axis_tdata,
  output wire                    m_axis_tlast,
  output wire                    m_axis_tuser,
  output wire                    prog_full_axis
);

  localparam integer TUSER_WIDTH = 1;
  localparam integer TID_WIDTH   = 1;
  localparam integer TDEST_WIDTH = 1;

  xpm_fifo_axis #(
    .CLOCKING_MODE     ("common_clock"),
    .FIFO_MEMORY_TYPE  (FIFO_MEMORY_TYPE),
    .PACKET_FIFO       (PACKET_FIFO),
    .FIFO_DEPTH        (FIFO_DEPTH),
    .TDATA_WIDTH       (TDATA_WIDTH),
    .TID_WIDTH         (TID_WIDTH),
    .TDEST_WIDTH       (TDEST_WIDTH),
    .TUSER_WIDTH       (TUSER_WIDTH),
    .PROG_FULL_THRESH  (PROG_FULL_THRESH)
  ) u_xpm_fifo_axis (
    .s_aresetn         (s_aresetn),
    .s_aclk            (s_aclk),
    .m_aclk            (m_aclk),
    .s_axis_tvalid     (s_axis_tvalid),
    .s_axis_tready     (s_axis_tready),
    .s_axis_tdata      (s_axis_tdata),
    .s_axis_tstrb      ({(TDATA_WIDTH/8){1'b1}}),
    .s_axis_tkeep      ({(TDATA_WIDTH/8){1'b1}}),
    .s_axis_tlast      (s_axis_tlast),
    .s_axis_tid        ({(TID_WIDTH){1'b0}}),
    .s_axis_tdest      ({(TDEST_WIDTH){1'b0}}),
    .s_axis_tuser      (s_axis_tuser),
    .m_axis_tvalid     (m_axis_tvalid),
    .m_axis_tready     (m_axis_tready),
    .m_axis_tdata      (m_axis_tdata),
    .m_axis_tstrb      (),
    .m_axis_tkeep      (),
    .m_axis_tlast      (m_axis_tlast),
    .m_axis_tid        (),
    .m_axis_tdest      (),
    .m_axis_tuser      (m_axis_tuser),
    .prog_full_axis    (prog_full_axis),
    .wr_data_count_axis(),
    .almost_full_axis  (),
    .prog_empty_axis   (),
    .rd_data_count_axis(),
    .almost_empty_axis (),
    .injectsbiterr_axis(1'b0),
    .injectdbiterr_axis(1'b0),
    .sbiterr_axis      (),
    .dbiterr_axis      ()
  );

endmodule
