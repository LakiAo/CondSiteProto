"""One explicit ordering for labels, ligand IDs, and molecular structures."""
LIGANDS = ("S_THIQ", "R_THIQ", "DHIQ")
TARGET_COLUMNS = ("log_g32_g0_ratio_1S-TIQ", "log_g32_g0_ratio_1R-TIQ", "log_g32_g0_ratio_1-DIQ")
ERROR_COLUMNS = tuple(column + "_err" for column in TARGET_COLUMNS)
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
