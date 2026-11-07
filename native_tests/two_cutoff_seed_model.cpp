// Deterministically activate every existing raw coordinate, including U/a and
// inner/joint heads. The output remains a normal serialized native model.
#include "dev_src/mtpr.h"
#include <cmath>
int main(int argc,char**argv){if(argc!=3)return 2;MLMTPR m(argv[1]);for(int p=0;p<m.CoeffCount();++p)m.Coeff()[p]+=.03*std::sin(p*.37+.2);m.LinCoeff();m.Save(argv[2]);return 0;}
