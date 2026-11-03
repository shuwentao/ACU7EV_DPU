module image_crop_core #(
    parameter PPC = 1 , //1,2
    parameter CPP = 2 , //1 VYUY,UYVY
    parameter BPP = 8 
)(
    input   logic                       clk                 ,
    input   logic                       rst                 ,
                                                             
    input   logic   [11:0]              width               ,
    input   logic   [11:0]              height              ,
    input   logic   [11:0]              x                   ,
    input   logic   [11:0]              y                   ,
    input   logic   [11:0]              w                   ,
    input   logic   [11:0]              h                   ,
                                                             
    input   logic                       s_axis_tuser        ,
    input   logic                       s_axis_tvalid       ,
    input   logic                       s_axis_tlast        ,
    input   logic   [PPC*CPP*BPP-1:0]   s_axis_tdata        ,
                                                         
    output  logic                       m_axis_tuser        ,
    output  logic                       m_axis_tvalid       ,
    output  logic                       m_axis_tlast        ,
    output  logic   [PPC*CPP*BPP-1:0]   m_axis_tdata         
);

logic [11:0] in_row  ;
logic [11:0] in_col  ;
logic [11:0] out_row ;
logic [11:0] out_col ;

logic [11:0] left_boundary   ;
logic [11:0] right_boundary  ;
logic [11:0] top_boundary    ;
logic [11:0] bottom_boundary ;


logic  pixel_in_region_left   ; 
logic  pixel_in_region_right  ; 
logic  pixel_in_region_top    ; 
logic  pixel_in_region_bottom ; 

logic  keep ;

logic s_axis_tvalid_d1 ;
logic s_axis_tvalid_d2 ;
logic s_axis_tvalid_d3 ;


logic [PPC*BPP*CPP-1:0] s_axis_tdata_d1 ;
logic [PPC*BPP*CPP-1:0] s_axis_tdata_d2 ;


always_ff@(posedge clk) begin
    if(rst)
        in_row <= 'b0 ;
    else if(s_axis_tvalid && s_axis_tuser)
        in_row <= 'b0 ;
    else if(s_axis_tvalid && s_axis_tlast)
        in_row <= in_row + 1 ;
end

always_ff@(posedge clk) begin
    if(rst)
        in_col <= 'b0 ;
    else if(s_axis_tvalid && s_axis_tlast)
        in_col <= 'b0 ;
    else if(s_axis_tvalid)
        in_col <= in_col + 1 ;
end

always_ff@(posedge clk) begin
    if(rst)
        out_row <= 'b0 ;
    else if(m_axis_tvalid && m_axis_tuser)
        out_row <= 'b0 ;
    else if(m_axis_tvalid && m_axis_tlast)
        out_row <= out_row + 1 ;
end

always_ff@(posedge clk) begin
    if(rst)
        out_col <= 'b0 ;
    else if(m_axis_tvalid && m_axis_tlast)
        out_col <= 'b0 ;
    else if(m_axis_tvalid)
        out_col <= out_col + 1 ;
end

always_ff@(posedge clk) begin
    left_boundary   <= x     ;
    right_boundary  <= x + w ;
    top_boundary    <= y     ;
    bottom_boundary <= y + h ;
end

always_ff@(posedge clk) begin
    pixel_in_region_left   <= in_col >= left_boundary   ;
    pixel_in_region_right  <= in_col <  right_boundary  ;
    pixel_in_region_top    <= in_row >= top_boundary    ; 
    pixel_in_region_bottom <= in_row <  bottom_boundary ;
end

always_ff@(posedge clk) begin
    keep <= pixel_in_region_left && pixel_in_region_right && pixel_in_region_top && pixel_in_region_bottom ;
end

always_ff@(posedge clk) begin
    s_axis_tvalid_d1 <= s_axis_tvalid    ;
    s_axis_tvalid_d2 <= s_axis_tvalid_d1 ;
    s_axis_tvalid_d3 <= s_axis_tvalid_d2 ;
end

always_ff@(posedge clk) begin
    s_axis_tdata_d1 <= s_axis_tdata    ;
    s_axis_tdata_d2 <= s_axis_tdata_d1 ;
end

always_ff@(posedge clk) begin
    if(keep)
        m_axis_tvalid <= s_axis_tvalid_d2 ;
    else
        m_axis_tvalid <= 1'b0 ;
end

always_ff@(posedge clk) begin
    if(keep) begin
        m_axis_tdata = s_axis_tdata_d2 ;
    end
end


assign m_axis_tuser = (out_row == 0) && (out_col == 0) ;
assign m_axis_tlast = (out_col == w - 1) ; 

endmodule
