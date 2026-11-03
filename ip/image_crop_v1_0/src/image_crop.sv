module image_crop #(
    parameter PPC = 1 , //1,2
    parameter CPP = 2 , //1 VYUY,UYVY
    parameter BPP = 8   //8,10,12
)(
    // AXI4-Lite bus (cpu control)
    input   logic                       S_AXI_ACLK      ,
    input   logic                       S_AXI_ARESETN   ,
    // - Write address                  
    input   logic                       S_AXI_AWVALID   ,
    output  logic                       S_AXI_AWREADY   ,
    input   logic   [31:0]              S_AXI_AWADDR    ,
    // - Write data
    input   logic                       S_AXI_WVALID    ,
    output  logic                       S_AXI_WREADY    ,
    input   logic   [31:0]              S_AXI_WDATA     ,
    input   logic   [3:0]               S_AXI_WSTRB     ,
    // - Write response
    output  logic                       S_AXI_BVALID    ,
    input   logic                       S_AXI_BREADY    ,
    output  logic   [1:0]               S_AXI_BRESP     ,
    // - Read address
    input   logic                       S_AXI_ARVALID   ,
    output  logic                       S_AXI_ARREADY   ,
    input   logic   [31:0]              S_AXI_ARADDR    ,
    // - Read data/response
    output  logic                       S_AXI_RVALID    ,
    input                               S_AXI_RREADY    ,
    output  logic   [31:0]              S_AXI_RDATA     ,
    output  logic   [1:0]               S_AXI_RRESP     ,

    input   logic                       s_axis_tuser    ,
    input   logic                       s_axis_tvalid   ,
    output  logic                       s_axis_tready   ,
    input   logic                       s_axis_tlast    ,
    input   logic   [PPC*CPP*BPP-1:0]   s_axis_tdata    ,

    output  logic                       m_axis_tuser    ,
    output  logic                       m_axis_tvalid   ,
    input   logic                       m_axis_tready   ,
    output  logic                       m_axis_tlast    ,
    output  logic   [PPC*CPP*BPP-1:0]   m_axis_tdata     
);


logic                   s_axis_tvalid_d1 ;

logic                   s_axis_tuser_d1  ;
logic                   s_axis_tlast_d1  ;
logic [PPC*BPP*CPP-1:0] s_axis_tdata_d1  ;

logic                   s_axis_tuser_d2  ;
logic                   s_axis_tlast_d2  ;
logic [PPC*BPP*CPP-1:0] s_axis_tdata_d2  ;

logic                   s_axis_tvalid_bypass ;
logic                   s_axis_tvalid_core ;

logic                   axis_tvalid  ;
logic                   axis_tuser   ;
logic                   axis_tlast   ;
logic [PPC*BPP*CPP-1:0] axis_tdata   ;

logic                   axis_tvalid_core  ;
logic                   axis_tuser_core   ;
logic                   axis_tlast_core   ;
logic [PPC*BPP*CPP-1:0] axis_tdata_core   ;



logic                   ip_core_reset ;
logic                   autostart     ; 
logic                   bypass        ; 
logic [11:0]            width         ; 
logic [11:0]            height        ; 
logic [11:0]            x             ; 
logic [11:0]            y             ; 
logic [11:0]            w             ; 
logic [11:0]            h             ; 


image_crop_axilite #(
    .PPC ( PPC ), 
    .CPP ( CPP ),
    .BPP ( BPP )
) u_image_crop_axilite(
    .S_AXI_ACLK                     ( S_AXI_ACLK                ),
    .S_AXI_ARESETN                  ( S_AXI_ARESETN             ),
    .S_AXI_AWVALID                  ( S_AXI_AWVALID             ),
    .S_AXI_AWREADY                  ( S_AXI_AWREADY             ),
    .S_AXI_AWADDR                   ( S_AXI_AWADDR              ),
    .S_AXI_WVALID                   ( S_AXI_WVALID              ),
    .S_AXI_WREADY                   ( S_AXI_WREADY              ),
    .S_AXI_WDATA                    ( S_AXI_WDATA               ),
    .S_AXI_WSTRB                    ( S_AXI_WSTRB               ),
    .S_AXI_BVALID                   ( S_AXI_BVALID              ),
    .S_AXI_BREADY                   ( S_AXI_BREADY              ),
    .S_AXI_BRESP                    ( S_AXI_BRESP               ),
    .S_AXI_ARVALID                  ( S_AXI_ARVALID             ),
    .S_AXI_ARREADY                  ( S_AXI_ARREADY             ),
    .S_AXI_ARADDR                   ( S_AXI_ARADDR              ),
    .S_AXI_RVALID                   ( S_AXI_RVALID              ),
    .S_AXI_RREADY                   ( S_AXI_RREADY              ),
    .S_AXI_RDATA                    ( S_AXI_RDATA               ),
    .S_AXI_RRESP                    ( S_AXI_RRESP               ),
    .s_axis_tvalid                  ( s_axis_tvalid             ),
    .s_axis_tready                  ( s_axis_tready             ),
    .s_axis_tuser                   ( s_axis_tuser              ),
    .ip_core_reset                  ( ip_core_reset             ),
    .autostart                      ( autostart                 ),
    .bypass                         ( bypass                    ),
    .width                          ( width                     ),
    .height                         ( height                    ),
    .x                              ( x                         ),
    .y                              ( y                         ),
    .w                              ( w                         ),
    .h                              ( h                         ) 
);

always_ff@(posedge S_AXI_ACLK) begin
    s_axis_tvalid_d1 <= s_axis_tvalid && s_axis_tready ;
    if(s_axis_tvalid && s_axis_tready) begin
        s_axis_tuser_d1 <= s_axis_tuser ;
        s_axis_tlast_d1 <= s_axis_tlast ;
        s_axis_tdata_d1 <= s_axis_tdata ;
    end
end

always_ff@(posedge S_AXI_ACLK) begin
    if(bypass) begin  
        s_axis_tvalid_core   <= 1'b0             ;
        s_axis_tvalid_bypass <= s_axis_tvalid_d1 ;
    end
    else begin
        s_axis_tvalid_core   <= s_axis_tvalid_d1 ;
        s_axis_tvalid_bypass <= 1'b0             ; 
    end
    s_axis_tuser_d2  <= s_axis_tuser_d1  ;
    s_axis_tlast_d2  <= s_axis_tlast_d1  ;
    s_axis_tdata_d2  <= s_axis_tdata_d1  ;
end

image_crop_core #(
    .PPC ( PPC ), //1,2
    .CPP ( CPP ), //1 VYUY,UYVY
    .BPP ( BPP ) 
) u_image_crop_core(
    .clk                            ( S_AXI_ACLK            ), 
    .rst                            ( ip_core_reset         ),
    .width                          ( width                 ),
    .height                         ( height                ),
    .x                              ( x                     ),
    .y                              ( y                     ),
    .w                              ( w                     ),
    .h                              ( h                     ),
    .s_axis_tuser                   ( s_axis_tuser_d2       ),
    .s_axis_tvalid                  ( s_axis_tvalid_core    ),
    .s_axis_tlast                   ( s_axis_tlast_d2       ),
    .s_axis_tdata                   ( s_axis_tdata_d2       ),
    .m_axis_tuser                   ( axis_tuser_core       ),
    .m_axis_tvalid                  ( axis_tvalid_core      ),
    .m_axis_tlast                   ( axis_tlast_core       ),
    .m_axis_tdata                   ( axis_tdata_core       ) 
);

always_ff@(posedge S_AXI_ACLK) begin
    if(bypass) begin
        axis_tvalid <= s_axis_tvalid_bypass ;
        axis_tuser  <= s_axis_tuser_d2      ;
        axis_tlast  <= s_axis_tlast_d2      ;
        axis_tdata  <= s_axis_tdata_d2      ;
    end
    else begin
        axis_tvalid <= axis_tvalid_core     ;
        axis_tuser  <= axis_tuser_core      ;
        axis_tlast  <= axis_tlast_core      ;
        axis_tdata  <= axis_tdata_core      ;
    end
end

xpm_axi_stream_fifo#(
     .TDATA_WIDTH       (((PPC*CPP*BPP+ 7)/8)*8),
    .PROG_FULL_THRESH   ( 16        ),
    .FIFO_DEPTH         ( 64        ),
    .PACKET_FIFO        ( "false"   ),
    .FIFO_MEMORY_TYPE   ( "auto"    ) 
)
u_xpm_axi_stream_fifo(
    .s_aclk                         (   S_AXI_ACLK                 ),
    .s_aresetn                      (   S_AXI_ARESETN              ),
    .m_aclk                         (   S_AXI_ACLK                 ),
                                          
    .s_axis_tvalid                  (   axis_tvalid                ),
    .s_axis_tready                  (                              ),
    .s_axis_tdata                   (   axis_tdata                 ),
    .s_axis_tlast                   (   axis_tlast                 ),
    .s_axis_tuser                   (   axis_tuser                 ),
                                          
    .m_axis_tvalid                  (   m_axis_tvalid              ),
    .m_axis_tready                  (   m_axis_tready              ),
    .m_axis_tdata                   (   m_axis_tdata               ),
    .m_axis_tlast                   (   m_axis_tlast               ),
    .m_axis_tuser                   (   m_axis_tuser               ),
    .prog_full_axis                 (   prog_full_axis             )
);

always_ff@(posedge S_AXI_ACLK) begin
    if(!S_AXI_ARESETN)
        s_axis_tready <= 1'b0 ;
    else if(~ autostart)
        s_axis_tready <= 1'b0 ;
    else
        s_axis_tready <= ~ prog_full_axis ;
end


endmodule
