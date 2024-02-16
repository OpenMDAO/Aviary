import aviary.api as av

AviaryAircraft = av.Aircraft
AviaryDynamic = av.Dynamic


# ---------------------------
# Aircraft data hierarchy
# ---------------------------

class Aircraft(AviaryAircraft):
	
	class Engine(AviaryAircraft.Engine):
		DESIGN_MACH = "aircraft:engine:design_mach"
		DESIGN_ALTITUDE = "aircraft:engine:design_alt"
		DESIGN_MASS_FLOW = "aircraft:engine:design_mass_flow"
		DESIGN_NET_THRUST = "aircraft:engine:design_net_thrust"  # will need to use this to calculate the Aviary-core variable aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST using number of engines


# ---------------------------
# Dynamics data hierarchy
# ---------------------------

class Dynamic(AviaryDynamic):
	
	class Engine:
		ELECTRIC_SHAFT_POWER = "dynamic:engine:electric_shaft_power"
		SHAFT_MECH_SPEED = "dynamic:engine:shaft_mech_speed" #Part power variable names
