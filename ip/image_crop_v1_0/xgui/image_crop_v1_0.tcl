# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BPP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CPP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PPC" -parent ${Page_0}


}

proc update_PARAM_VALUE.BPP { PARAM_VALUE.BPP } {
	# Procedure called to update BPP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BPP { PARAM_VALUE.BPP } {
	# Procedure called to validate BPP
	return true
}

proc update_PARAM_VALUE.CPP { PARAM_VALUE.CPP } {
	# Procedure called to update CPP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CPP { PARAM_VALUE.CPP } {
	# Procedure called to validate CPP
	return true
}

proc update_PARAM_VALUE.PPC { PARAM_VALUE.PPC } {
	# Procedure called to update PPC when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PPC { PARAM_VALUE.PPC } {
	# Procedure called to validate PPC
	return true
}


proc update_MODELPARAM_VALUE.PPC { MODELPARAM_VALUE.PPC PARAM_VALUE.PPC } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PPC}] ${MODELPARAM_VALUE.PPC}
}

proc update_MODELPARAM_VALUE.CPP { MODELPARAM_VALUE.CPP PARAM_VALUE.CPP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CPP}] ${MODELPARAM_VALUE.CPP}
}

proc update_MODELPARAM_VALUE.BPP { MODELPARAM_VALUE.BPP PARAM_VALUE.BPP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BPP}] ${MODELPARAM_VALUE.BPP}
}

