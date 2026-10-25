// Isolated checker-control self-test. Deliberate mismatches must not qualify.
#define main radial_full_qualification_main
#include "radial_test_device.cpp"
#undef main
int main() {
  collect_numerical_failures=true;
  check(1.,0.,"deliberate value mismatch");
  check(2.,0.,"deliberate derivative mismatch");
  check(std::numeric_limits<double>::quiet_NaN(),0.,"deliberate nonfinite mismatch");
  check(std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),"matching NaN");
  if(numerical_failures!=3)return 1;
  collect_numerical_failures=false;bool stopped=false;
  try{check(1.,0.,"normal mode must stop");}catch(const NumericalFailure&){stopped=true;}
  if(!stopped||numerical_failures!=4)return 1;
  // Collection cannot catch/misclassify an ordinary runtime exception.
  bool fatal=false;
  try{throw std::runtime_error("runtime sentinel");}
  catch(const NumericalFailure&){return 1;}
  catch(const std::runtime_error&){fatal=true;}
  // Run every real case after the deliberate comparison failures. Its summary
  // must stay FAIL and its qualification return code must remain 1.
  int argc=1;char name[]="collection-selftest";char*argv[]={name,nullptr};
  const int qualification_rc=radial_full_qualification_main(argc,argv);
  if(!fatal||qualification_rc!=1||direct_cases!=146||table_cases!=87||fallback_cases!=1)return 1;
  std::cout<<"COLLECTION_CONTROL_SELFTEST_PASS (four deliberate numeric failures; full qualification correctly FAIL/exit1)\n";
  return 0;
}
