#pragma once

#include <mlir/Pass/Pass.h>

namespace colgm_mlir {

std::unique_ptr<mlir::Pass> create_scf_for_rank0_scalar_pass();

}
