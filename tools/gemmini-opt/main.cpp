// Thin gemmini-opt driver emitted by merlin-target-publish (WS-E).
//
// The buildable OOT tree at the repo root IS the published target repo; the real codegen
// payload rides under payload/. This driver accepts the experiment-ABI entrypoint flags so
// the repo is contract-shaped and merlin.targetgen.oot_runner can build it on a fresh clone.
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--version") == 0) {
      std::printf("gemmini-opt (merlin publish skeleton)\n");
      return 0;
    }
  }
  std::fprintf(stderr, "gemmini-opt: thin publish-skeleton driver; "
               "see payload/ for the codegen artifact.\n");
  return 0;
}
