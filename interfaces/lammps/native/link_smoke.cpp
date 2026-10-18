#include "sus2-native/dev_src/mtpr.h"
int main(int argc, char **argv)
{
  Configuration cfg;
  Neighborhoods neighborhoods;
  // A default-constructed legacy MLMTPR has no allocated model and its
  // historical destructor expects Load() first. Keep the model branch in the
  // link test without invoking that unsupported lifecycle on the no-file run.
  if (argc > 1) {
    MLMTPR model(argv[1]);
    return model.HasCompleteParameters() ? 0 : 1;
  }
  return cfg.size() + neighborhoods.size();
}
