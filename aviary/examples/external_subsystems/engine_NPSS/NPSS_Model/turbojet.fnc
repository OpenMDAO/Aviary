real MaxThrust, TargetThrust;
real W_DES, Alt_DES, MN_DES;
real thrust_training_data[];
real fuelflow_training_data[];
real thrustmax_training_data[];

Independent ind_W { varName = "start.W_in"; } 
Independent ind_FAR { varName = "burner.FAR"; } 

Dependent dep_Fn { eq_lhs = "nozzle.Fg-inlet.Fram"; eq_rhs = "25000."; }
Dependent dep_T4 { eq_lhs = "burner.Fl_O.Tt"; eq_rhs = "6*inlet.Fl_O.Tt"; }
Dependent Max_NcHPC { eq_lhs = "HPC.S_map.NcMap"; eq_rhs = "1.00"; }
Dependent Target_Fnet { eq_lhs = "PERF.Fn"; eq_rhs = "TargetThrust";}

void update_training_data(){
  thrust_training_data.append(PERF.Fn);
  fuelflow_training_data.append(PERF.Wfuel);
  thrustmax_training_data.append(MaxThrust);
}

void RunMaxPower() { 
   autoSolverSetup(); 
   solver.addIndependent( "ind_FAR" );
   solver.addDependent( "Max_NcHPC" );

   run(); 
   MaxThrust = PERF.Fn;
}

void RunPartPower(){
  // Run to part power, PERF.PC must be defined outside this function call
  autoSolverSetup();
  solver.addIndependent( "ind_FAR" );
  TargetThrust = PERF.PercentPower(PERF.PC)*MaxThrust;
  solver.addDependent( "Target_Fnet" );
  run();
}

void RunThrottleHook(real RunMN, real RunAlt){
  ambient.MN_in  = RunMN;
  ambient.alt_in = RunAlt;
  cout << "Run ThrottleHook at: MN = " << RunMN << "      Alt = "<< RunAlt;

  PERF.PC = 50.0; ++CASE; RunMaxPower(); cout << " Pwr :50.. "; Aviarysheet.update(); update_training_data(); //page.display(); 
  PERF.PC = 47.0; ++CASE; RunPartPower(); cout << "47.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 44.0; ++CASE; RunPartPower(); cout << "44.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 41.0; ++CASE; RunPartPower(); cout << "41.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 38.0; ++CASE; RunPartPower(); cout << "38.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 35.0; ++CASE; RunPartPower(); cout << "35.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 32.0; ++CASE; RunPartPower(); cout << "32.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 26.0; ++CASE; RunPartPower(); cout << "26.. "; Aviarysheet.update(); update_training_data(); //page.display();
  PERF.PC = 21.0; ++CASE; RunPartPower(); cout << "21.. "; Aviarysheet.update(); update_training_data(); //page.display();
  cout<<endl;

  // run back to maximum power for convergence purposes
  PERF.PC = 32.0; RunPartPower();
  PERF.PC = 44.0; RunPartPower();
  PERF.PC = 50.0; RunMaxPower();

}