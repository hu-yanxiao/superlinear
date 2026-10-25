// Qualification-only launch recorder. No CUDA, MPI, fence or model calls.
#include "kokkos_role_trace_abi.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>

namespace {
std::mutex lock;
std::FILE *output=nullptr;
std::uint64_t sequence=0;
void open_output() {
  if(output)return;
  const char *directory=std::getenv("SUS2_ROLE_TRACE_DIR");
  if(!directory)return;
  char name[4096];
  std::snprintf(name,sizeof(name),"%s/trace.%ld.jsonl",directory,long(getpid()));
  output=std::fopen(name,"a");
  if(!output)std::fprintf(stderr,"SUS2_ROLE_TRACE_ERROR cannot open %s\n",name);
}
void flush() {if(output)std::fflush(output);}
void string(const char *value) {
  std::fputc('"',output);
  for(const unsigned char *p=reinterpret_cast<const unsigned char*>(value);*p;++p){
    if(*p=='"'||*p=='\\')std::fputc('\\',output);
    if(*p<32)std::fprintf(output,"\\u%04x",unsigned(*p));else std::fputc(*p,output);
  }
  std::fputc('"',output);
}
}
extern "C" void kokkosp_init_library(const int,const std::uint64_t version,
    const std::uint32_t,Kokkos_Profiling_KokkosPDeviceInfo*) {
  std::lock_guard<std::mutex> guard(lock);open_output();
  if(output)std::fprintf(output,"{\"event\":\"init\",\"interface_version\":%llu,\"pid\":%ld,\"qualification_only\":true}\n",
    static_cast<unsigned long long>(version),long(getpid()));flush();
}
extern "C" void kokkosp_request_tool_settings(const std::uint32_t version,
    Kokkos_Tools_ToolSettings *settings) {
  // Kokkos defaults to true. Explicitly decline automatic global fencing.
  settings->requires_global_fencing=false;
  std::lock_guard<std::mutex> guard(lock);open_output();
  if(output)std::fprintf(output,"{\"event\":\"settings\",\"version\":%u,\"requires_global_fencing\":false}\n",version);
  flush();
}
extern "C" void kokkosp_begin_parallel_for(const char *name,const std::uint32_t device,
    std::uint64_t *identifier) {
  *identifier=0;
  if(std::strncmp(name,"sus2_",5)!=0)return;
  std::lock_guard<std::mutex> guard(lock);open_output();*identifier=++sequence;
  if(output){std::fprintf(output,"{\"event\":\"begin\",\"id\":%llu,\"device_id\":%u,\"device_type\":%u,\"name\":",
      static_cast<unsigned long long>(*identifier),device,device>>24);
    string(name);std::fputs("}\n",output);flush();}
}
extern "C" void kokkosp_end_parallel_for(std::uint64_t identifier) {
  if(!identifier)return;
  std::lock_guard<std::mutex> guard(lock);
  if(output)std::fprintf(output,"{\"event\":\"end\",\"id\":%llu}\n",static_cast<unsigned long long>(identifier));flush();
}
extern "C" void kokkosp_finalize_library() {
  std::lock_guard<std::mutex> guard(lock);
  if(output){std::fputs("{\"event\":\"finalize\"}\n",output);flush();std::fclose(output);output=nullptr;}
}
