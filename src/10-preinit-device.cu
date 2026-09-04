#include "00-common.h"

int main(int argc, char **argv) {
  int early_device = preselect_device_from_env();
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int final_device = select_device(MPI_COMM_WORLD, &local_rank);
  bool ok = early_device < 0 || early_device == final_device;
  std::printf(
      "rank %d local rank %d: before MPI_Init GPU %d, final GPU %d %s\n", rank,
      local_rank, early_device, final_device,
      early_device < 0 ? "launcher variable unavailable"
      : ok             ? "PASS"
                       : "FAIL");
  MPI_Finalize();
  return ok ? 0 : 1;
}
