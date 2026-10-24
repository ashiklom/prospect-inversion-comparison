# Compile all Stan models once (before parallel fits share the executables)
library(cmdstanr)
set_cmdstan_path(file.path(Sys.getenv("CONDA_PREFIX"), "bin", "cmdstan"))
dir.create("stan/build", showWarnings = FALSE)
for (m in c("iid", "hetero_ar1")) {
  cmdstan_model(sprintf("stan/prospect_%s.stan", m), include_paths = "stan",
                user_header = normalizePath("stan/expint.hpp"), dir = "stan/build")
}
