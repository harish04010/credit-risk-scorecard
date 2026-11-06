# Independent validation of the Python scorecard, in base R.
#
# A model risk team doesn't take the developer's numbers on trust. This script
# refits the logistic regression on the same WoE-transformed development data
# with R's glm(), and recomputes AUC and KS on the holdout from the raw scores
# with its own code. It stops with an error if anything disagrees.
#
#   Rscript r/validate_scorecard.R

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))
res <- file.path(here, "..", "results")

dev <- read.csv(file.path(res, "r_dev_woe.csv"))
py <- read.csv(file.path(res, "py_coefficients.csv"))
hold <- read.csv(file.path(res, "r_holdout_scores.csv"))

# 1. Refit the model.
form <- as.formula(paste("default ~", paste(py$feature, collapse = " + ")))
fit <- glm(form, data = dev, family = binomial())
r_coef <- coef(fit)[py$feature]
diff_coef <- max(abs(r_coef - py$coefficient))

# 2. AUC by the Mann-Whitney rank formula. Low score = high risk, so rank on -score.
risk <- -hold$score
r <- rank(risk)
n1 <- sum(hold$default == 1); n0 <- sum(hold$default == 0)
auc <- (sum(r[hold$default == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)

# 3. KS: the largest gap between the empirical score distributions of bads and goods.
grid <- sort(unique(hold$score))
F_bad <- ecdf(hold$score[hold$default == 1])(grid)
F_good <- ecdf(hold$score[hold$default == 0])(grid)
ks <- max(abs(F_bad - F_good))

# 4. Significance of each coefficient, which Python's sklearn doesn't report.
z <- summary(fit)$coefficients[py$feature, "z value"]

pym <- read.csv(file.path(res, "py_holdout_metrics.csv"))
py_auc <- pym$value[pym$metric == "auc"]
py_ks <- pym$value[pym$metric == "ks"]

checks <- c(
  coefficients_match = diff_coef < 1e-4,
  auc_match = abs(auc - py_auc) < 1e-6,
  ks_match = abs(ks - py_ks) < 1e-6,
  all_coefficients_significant = all(abs(z) > 1.96),
  all_signs_negative = all(r_coef < 0)
)

cat(sprintf("Coefficients: largest difference from Python %.2e\n", diff_coef))
cat(sprintf("Holdout AUC:  R %.6f   Python %.6f\n", auc, py_auc))
cat(sprintf("Holdout KS:   R %.6f   Python %.6f\n", ks, py_ks))
cat(sprintf("Smallest |z| of any coefficient: %.1f\n", min(abs(z))))
for (k in names(checks)) cat(sprintf("  %-30s %s\n", k, if (checks[[k]]) "pass" else "FAIL"))

json <- sprintf('{"max_coefficient_difference": %.3e, "auc_r": %.6f, "auc_python": %.6f, "ks_r": %.6f, "ks_python": %.6f, "min_abs_z": %.2f, "checks": {%s}}',
                diff_coef, auc, py_auc, ks, py_ks, min(abs(z)),
                paste(sprintf('"%s": %s', names(checks), tolower(as.character(checks))), collapse = ", "))
writeLines(json, file.path(res, "r_validation.json"))
if (!all(checks)) stop("Independent validation failed")
