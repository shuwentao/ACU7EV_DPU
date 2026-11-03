module image_crop_axilite #(
    parameter PPC = 1 , //1,2
    parameter CPP = 2 , //1 VYUY,UYVY
    parameter BPP = 8   //8,10,12
)(
    // AXI4-Lite bus (cpu control)
    input   logic                           S_AXI_ACLK      ,
    input   logic                           S_AXI_ARESETN   ,
    // - Write address                      
    input   logic                           S_AXI_AWVALID   ,
    output  logic                           S_AXI_AWREADY   ,
    input   logic   [31:0]                  S_AXI_AWADDR    ,
    // - Write data
    input   logic                           S_AXI_WVALID    ,
    output  logic                           S_AXI_WREADY    ,
    input   logic   [31:0]                  S_AXI_WDATA     ,
    input   logic   [3:0]                   S_AXI_WSTRB     ,
    // - Write response
    output  logic                           S_AXI_BVALID    ,
    input   logic                           S_AXI_BREADY    ,
    output  logic   [1:0]                   S_AXI_BRESP     ,
    // - Read address
    input   logic                           S_AXI_ARVALID   ,
    output  logic                           S_AXI_ARREADY   ,
    input   logic   [31:0]                  S_AXI_ARADDR    ,
    // - Read data/response
    output  logic                           S_AXI_RVALID    ,
    input                                   S_AXI_RREADY    ,
    output  logic   [31:0]                  S_AXI_RDATA     ,
    output  logic   [1:0]                   S_AXI_RRESP     ,
    
    input   logic                           s_axis_tvalid   ,
    input   logic                           s_axis_tready   ,
    //input   logic                           s_axis_tlast    ,
    input   logic                           s_axis_tuser    ,
    
    //input   logic                           m_axis_tvalid   ,
    //input   logic                           m_axis_tready   ,
    //input   logic                           m_axis_tlast    ,
    //input   logic                           m_axis_tuser    ,
    
    output  logic                           ip_core_reset   , 

    output  logic                           autostart       ,
    output  logic                           bypass          ,
    output  logic   [11:0]                  width           ,
    output  logic   [11:0]                  height          ,
    output  logic   [11:0]                  x               ,
    output  logic   [11:0]                  y               ,
    output  logic   [11:0]                  w               ,
    output  logic   [11:0]                  h                
);


//
localparam IP_VERSION        = 32'h0 ;

// AXI Bus responses
localparam AXI4_RESP_OKAY    = 2'b00; // Okay
localparam AXI4_RESP_SLVERR  = 2'b10; // Slave error

// Register map addresses
localparam CONTROL_ADDR      = 12'h00 ;
localparam IP_VERSION_ADDR   = 12'h10 ;
localparam CFG_VERSION_ADDR  = 12'h14 ;
localparam BYPASS_ADDR       = 12'h18 ;
localparam WIDTH_ADDR        = 12'h20 ;
localparam HEIGHT_ADDR       = 12'h24 ;
localparam X_ADDR            = 12'h28 ;   
localparam Y_ADDR            = 12'h2C ;   
localparam W_ADDR            = 12'h30 ;   
localparam H_ADDR            = 12'h34 ;   


localparam S_WR_RST  = 2'd0 ;
localparam S_WR_ADDR = 2'd1 ;
localparam S_WR_DATA = 2'd2 ;
localparam S_WR_RESP = 2'd3 ;

localparam S_RD_RST  = 2'd0 ;
localparam S_RD_ADDR = 2'd1 ;
localparam S_RD_DATA = 2'd2 ;
localparam S_RD_RESP = 2'd3 ;

logic [1:0]     wr_cstate ;
logic [1:0]     wr_nstate ;
logic [11:0]    wrAddr;
logic [1:0]     rd_cstate ;
logic [1:0]     rd_nstate ;
logic [11:0]    rdAddr;
logic           ReadAddrNOK;
logic [31:0]    rdData ;

logic           bypass_bus ;
logic [11:0]    width_bus  ;
logic [11:0]    height_bus ;
logic [11:0]    x_bus      ;
logic [11:0]    y_bus      ;
logic [11:0]    w_bus      ;
logic [11:0]    h_bus      ;


logic           bypass_work;
logic [11:0]    width_work ;
logic [11:0]    height_work;
logic [11:0]    x_work     ;
logic [11:0]    y_work     ;
logic [11:0]    w_work     ;
logic [11:0]    h_work     ;

logic           sof        ;
logic           cfg_version;
logic           cfg_version_latch;

always_ff@(posedge S_AXI_ACLK) begin
  if (!S_AXI_ARESETN) begin
    S_AXI_AWREADY       <= 1'b0;
    S_AXI_WREADY        <= 1'b0;
    S_AXI_BVALID        <= 1'b0;
    wrAddr              <=  'h0;
    wr_cstate           <= S_WR_RST;
  end
  else begin
    case (wr_cstate)
      S_WR_RST : begin
        S_AXI_AWREADY   <= 1'b1;
        S_AXI_WREADY    <= 1'b0;
        S_AXI_BVALID    <= 1'b0;
        wr_cstate       <= S_WR_ADDR;
      end

      S_WR_ADDR : begin
        S_AXI_AWREADY   <= 1'b1;
        if (S_AXI_AWVALID) begin
          S_AXI_AWREADY <= 1'b0;
          S_AXI_WREADY  <= 1'b1;
          wrAddr        <= S_AXI_AWADDR[11:0] ;
          wr_cstate     <= S_WR_DATA;
        end
      end

      S_WR_DATA : begin
        S_AXI_WREADY    <= 1'b1;
        if (S_AXI_WVALID) begin
          S_AXI_WREADY  <= 1'b0;
          S_AXI_BVALID  <= 1'b1;
          wr_cstate     <= S_WR_RESP;
        end
      end

      S_WR_RESP : begin
        S_AXI_BVALID    <= 1'b1;
        if (S_AXI_BREADY) begin
          S_AXI_BVALID  <= 1'b0;
          wr_cstate     <= S_WR_RST;
        end
      end

      default : wr_cstate <= S_WR_RST;
    endcase
  end
end

// Write address decoder
always_ff@(posedge S_AXI_ACLK) begin
  if (!S_AXI_ARESETN) begin
    S_AXI_BRESP <= AXI4_RESP_OKAY;
    
    autostart   <=  1'b0  ;
    cfg_version <=  1'b0  ;
    bypass_bus  <=  1'b0  ;
    width_bus   <=  12'b0 ;
    height_bus  <=  12'b0 ;
    x_bus       <=  12'b0 ;
    y_bus       <=  12'b0 ;
    w_bus       <=  12'b0 ;
    h_bus       <=  12'b0 ;
  end
  else begin
    if (S_AXI_WREADY && S_AXI_WVALID) begin
      S_AXI_BRESP     <= AXI4_RESP_OKAY;
      case (wrAddr)
        CONTROL_ADDR     : autostart   <= S_AXI_WDATA[4]  ;
        CFG_VERSION_ADDR : cfg_version <= S_AXI_WDATA[0] ;
        BYPASS_ADDR      : bypass_bus  <= S_AXI_WDATA[0] ;
        WIDTH_ADDR       : width_bus   <= S_AXI_WDATA[11:0] ; 
        HEIGHT_ADDR      : height_bus  <= S_AXI_WDATA[11:0] ; 
        X_ADDR           : x_bus <= S_AXI_WDATA[11:0] ; 
        Y_ADDR           : y_bus <= S_AXI_WDATA[11:0] ; 
        W_ADDR           : w_bus <= S_AXI_WDATA[11:0] ; 
        H_ADDR           : h_bus <= S_AXI_WDATA[11:0] ; 
        default : S_AXI_BRESP <= AXI4_RESP_SLVERR;
      endcase
    end
  end
end

always_ff@(posedge S_AXI_ACLK) begin
    if(!S_AXI_ARESETN) begin
        bypass_work       <= 1'b1       ;
        width_work        <= 12'd1280   ;
        height_work       <= 12'd720    ;
        x_work            <= 12'd0      ;
        y_work            <= 12'd0      ;
        w_work            <= 12'd1280   ;
        h_work            <= 12'd720    ;
        cfg_version_latch <= 1'b0       ;
    end
    else if(s_axis_tvalid && s_axis_tready && s_axis_tuser && (cfg_version != cfg_version_latch)) begin
        bypass_work       <= bypass_bus ;
        width_work        <= width_bus  ;
        height_work       <= height_bus ;
        x_work            <= x_bus      ;
        y_work            <= y_bus      ;
        w_work            <= w_bus      ;
        h_work            <= h_bus      ;
        cfg_version_latch <= cfg_version ;
    end
end


assign sof = s_axis_tvalid && s_axis_tready && s_axis_tuser ;

always_ff @(posedge S_AXI_ACLK) begin
    ip_core_reset <= (~ S_AXI_ARESETN) ||  sof  || (~ autostart);
end

assign bypass = bypass_work ;
assign width  = width_work  ;
assign height = height_work ;
assign x      = x_work      ;
assign y      = y_work      ;
assign w      = w_work      ;
assign h      = h_work      ;

////////////////////////////////////////////////////////
// Read channel

// Statemachine for taking care of the read signals
always@(posedge S_AXI_ACLK) begin
  if (!S_AXI_ARESETN) begin
    S_AXI_ARREADY       <= 1'b0;
    S_AXI_RRESP         <= AXI4_RESP_OKAY;
    S_AXI_RVALID        <= 1'b0;
    S_AXI_RDATA         <=  'h0;
    rdAddr              <=  'h0;
    rd_cstate           <= S_RD_RST ;
  end
  else begin
    case (rd_cstate)
      S_RD_RST :
      begin
        S_AXI_ARREADY   <= 1'b1;
        S_AXI_RRESP     <= AXI4_RESP_OKAY;
        S_AXI_RVALID    <= 1'b0;
        S_AXI_RDATA     <=  'h0;
        rdAddr          <=  'h0;
        rd_cstate       <= S_RD_ADDR;
      end

      S_RD_ADDR :
      begin
        S_AXI_ARREADY   <= 1'b1;
        if (S_AXI_ARVALID)
        begin
          S_AXI_ARREADY <= 1'b0;
          rdAddr        <= S_AXI_ARADDR[11:0];
          rd_cstate     <= S_RD_DATA;
        end
      end

      S_RD_DATA :
      begin
        if (ReadAddrNOK)
          S_AXI_RRESP   <= AXI4_RESP_SLVERR;
        else
          S_AXI_RRESP   <= AXI4_RESP_OKAY;

        S_AXI_RDATA     <= rdData;
        S_AXI_RVALID    <= 1'b1;
        rd_cstate       <= S_RD_RESP;
      end

      S_RD_RESP :
      begin
        S_AXI_RVALID    <= 1'b1;
        if (S_AXI_RREADY)
        begin
          S_AXI_RVALID  <= 1'b0;
          rd_cstate     <= S_RD_RST;
        end
      end

      default : rd_cstate <= S_RD_RST;
    endcase
  end
end

// Read address decoder
always@(*) begin
  ReadAddrNOK   = 1'b0;
  rdData        =  'h0;
  case (rdAddr)
        CONTROL_ADDR     : rdData = {27'b0,autostart,4'b0} ;
        IP_VERSION_ADDR  : rdData = IP_VERSION             ;
        CFG_VERSION_ADDR : rdData = {31'b0,cfg_version}    ;
        BYPASS_ADDR      : rdData = {31'b0,bypass_bus}     ;
        WIDTH_ADDR       : rdData = {20'b0,width_bus}      ;
        HEIGHT_ADDR      : rdData = {20'b0,height_bus}     ;
        X_ADDR           : rdData = {20'b0,x_bus}          ;
        Y_ADDR           : rdData = {20'b0,y_bus}          ;
        W_ADDR           : rdData = {20'b0,w_bus}          ;
        H_ADDR           : rdData = {20'b0,h_bus}          ;
    default : ReadAddrNOK  = 1'b1;
  endcase
end


endmodule


