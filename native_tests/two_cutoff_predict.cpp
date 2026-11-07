// Full precision native oracle bridge; ordinary CFG Save rounds force/stress.
#include "dev_src/mtpr.h"
#include <fstream>
#include <iomanip>
int main(int argc,char**argv){if(argc!=4)return 2;MLMTPR m(argv[1]);std::ifstream in(argv[2]);std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"records\":[";Configuration c;int k=0;
while(c.Load(in)){m.CalcEFS(c);if(k)out<<',';out<<"{\"config\":"<<k++<<",\"energy\":"<<c.energy<<",\"forces\":[";for(int i=0;i<c.size();++i){if(i)out<<',';out<<'[';for(int a=0;a<3;++a){if(a)out<<',';out<<c.force(i,a);}out<<']';}out<<"],\"plus_stress\":[";for(int a=0;a<3;++a){if(a)out<<',';out<<'[';for(int b=0;b<3;++b){if(b)out<<',';out<<c.stresses[a][b];}out<<']';}out<<"]}";}out<<"]}\n";return k?0:1;}
