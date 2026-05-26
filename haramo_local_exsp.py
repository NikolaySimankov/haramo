#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd

from pathlib import Path

from itertools import product

import os
import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from haramo.classification import magic_now

#############
# Functions #
#############


def vectors_to_matrix(*vectors):
    return [list(item) for item in product(*vectors)]


########
# Main #
########

if __name__ == "__main__":

    path = Path(".")

    output_dir = path / "local_test"
    output_dir.mkdir(exist_ok=True)

    data = Path("C:\\Users\\nikol\\Documents\\GitHub\\PhytovirusDB\\data")
    data.mkdir(exist_ok=True)

    features = Path("C:\\Users\\nikol\\Documents\\GitHub\\PhytovirusDB\\data\\features")
    features.mkdir(exist_ok=True)

    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)

    # Load the target file into a DataFrame
    all_targets = pd.read_csv(
        data / "host_species_confidence.tsv", sep="\t", index_col="Virus_Species"
    )

    target_counts = all_targets.sum(skipna=True)
    consistent_targets = target_counts[target_counts >= 12].index
    all_targets = all_targets[consistent_targets]
    all_targets.reset_index(inplace=True)

    # Load the feature DataFrames
    feature_files = {
        "ctd": "X_ctd.tsv",
        # "ctdc": "X_ctdc.tsv",
        # "ctdt": "X_ctdt.tsv",
        # "ctdd": "X_ctdd.tsv",
        # "aac": "X_aac.tsv",
        # "b2b": "X_b2btools.tsv",
        "nsp": "X_netsurfp.tsv",
        # "residue": "X_residue.tsv",
        # "biophys": "X_biophys.tsv",
        "class": "X_class.tsv",
    }

    all_X = {
        name: pd.read_csv(features / fname, sep="\t", index_col="Prot_ID")
        for name, fname in feature_files.items()
    }

    # Load the Protein metadata file into a DataFrame
    taxo = pd.read_csv(
        data / "clustered_proteins_V5.2.tsv", sep="\t", index_col="Prot_ID"
    )

    # Restrict metadata to proteins present in every feature dataset and with a known name
    common_index = taxo.index
    for X in all_X.values():
        common_index = common_index.intersection(X.index)
    taxo = taxo.loc[common_index]
    taxo = taxo[taxo["Definitive_name"].notna()]

    # Split comma-separated values and remove duplicates
    proteins = [
        # "DNA replication protein",
        # "RNA-dependent RNA polymerase",
        # "DNA-RNA polymerase superfamily",
        # "Reverse transcriptase",
        # "Coat protein",
        # "Movement protein",
        # "Transactivator-Viroplasmin protein",
        "RNA silencing suppressor",
        # "Vector transmission protein",
        # "RNA-dependent RNA polymerase complex",
        # "Reverse transcriptase complex",
        # "Glycoprotein",
    ]

    for protein in proteins:

        # forbid matches where the phrase is followed by another word (e.g. "... complex")
        pattern = rf"\b{re.escape(protein)}\b(?!\s+\w)"
        biophys = taxo.loc[
            taxo["Definitive_name"].str.contains(
                pattern, case=False, regex=True, na=False
            )
        ]
        biophys = biophys.reset_index()[["Prot_ID", "Virus_Species"]]

        if len(biophys) >= 500:

            intersect = pd.merge(biophys, all_targets, how="inner", on="Virus_Species")

            prot_ids = intersect["Prot_ID"]

            # Align each feature dataset to the intersected protein IDs; fill all-NaN values by 0
            datasets = {name: X.loc[prot_ids].fillna(0) for name, X in all_X.items()}

            targets = intersect[["Prot_ID"] + list(all_targets.columns)]
            targets.drop(columns=["Virus_Species"], inplace=True)
            targets.set_index("Prot_ID", inplace=True)

            groups = intersect.set_index("Prot_ID")["Virus_Species"]

            prot_counts = targets.apply(lambda x: x.sum(), axis=0).sort_values(
                ascending=False
            )
            consistant_targets = prot_counts[prot_counts >= 100].index
            # chose 2 random consistant targets for each protein

            target = "Gomphrena globosa"
            # for target in consistant_targets:

            y = targets[target].dropna()
            groups = groups.loc[y.index]
            datasets = {name: X.loc[y.index] for name, X in datasets.items()}

            magic_now(
                X=datasets,
                y=y,
                outer_cv_groups=groups,
                inner_cv_groups=groups,
                scoring="FNFP Loss",
                algorithm=["LGBM"],
                scaler="robust",
                feature_selector="pvalue",
                hyperparameters="optimize",
                n_trials=10,
                output_dir=output_dir,
                plots=True,
                tag=f"_{protein}_{target}",
                n_jobs=12,
                calibration="auto",
                optimize_threshold=True,
                pos_weight_factor=1.5,
            )
