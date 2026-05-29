#!/usr/bin/env python

###########
# Imports #
###########

import pandas as pd

from pathlib import Path

from dawgz import (
    job,
    ensure,
    schedule,
)

from itertools import product

import os
import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from haramo.classification import magic_now

from sklearn.metrics import make_scorer, matthews_corrcoef

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
)

#############
# Functions #
#############


def get_parser():

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--d",
        dest="folder",
        help="destination folder",
        required=True,
    )

    parser.add_argument(
        "--t",
        default=50,
        type=int,
        dest="n_trials",
        help="number of trials for the hyperparameter optimization process",
        required=False,
    )

    parser.add_argument(
        "--v",
        default=1,
        type=int,
        dest="verbose",
        help="whether to print the progress of the training and validation",
        required=False,
    )

    parser.add_argument(
        "--b",
        default="slurm",
        type=str,
        dest="backend",
        help="backend to use for the parallelization of the jobs slurm, async or dummy",
        required=True,
    )

    return parser


def vectors_to_matrix(*vectors):
    return [list(item) for item in product(*vectors)]


########
# Main #
########

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    mcc_scorer = make_scorer(matthews_corrcoef)

    path = Path(".")

    output_dir = path / args.folder
    output_dir.mkdir(exist_ok=True)

    data = path / "data"
    data.mkdir(exist_ok=True)

    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)

    jobs = []

    # Load the target file into a DataFrame
    all_targets = pd.read_csv(data / "xxxxx.tsv", sep="\t", index_col="Virus_Species")

    target_counts = all_targets.sum(skipna=True)
    consistent_targets = target_counts[target_counts >= 12].index
    all_targets = all_targets[consistent_targets]
    all_targets.reset_index(inplace=True)

    protein = "Movement protein"

    algos = ["LGBM", "CatB", "XGB", "MLP"]

    # Load the feature DataFrames
    feature_files = {
        "ctd": "X_ctd.tsv",
        "ctdc": "X_ctdc.tsv",
        "ctdt": "X_ctdt.tsv",
        "ctdd": "X_ctdd.tsv",
        "NSP": "X_netsurfp.tsv",
        "biophys": "X_biophys.tsv",
        "class": "X_class.tsv",
        "XXX": "xxxxxxx.tsv",
    }

    all_X = {
        name: pd.read_csv(data / fname, sep="\t", index_col="Prot_ID")
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

    # forbid matches where the phrase is followed by another word (e.g. "... complex")
    pattern = rf"\b{re.escape(protein)}\b(?!\s+\w)"
    biophys = taxo.loc[
        taxo["Definitive_name"].str.contains(pattern, case=False, regex=True, na=False)
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

        for algorithm in algos:

            kwargs_heavy = {"cpus": 12, "ram": "64GB", "time": "24:00:00"}

            @job(name=f"{args.folder}:{algorithm}_sp100", **kwargs_heavy)
            def optimisation():

                for target in consistant_targets:
                    log_path = logs / f"{args.folder}_{algorithm}_{target}.log"
                    if not os.path.exists(log_path):
                        y = targets[target].dropna()
                        datasets = {
                            name: X.loc[y.index] for name, X in datasets.items()
                        }

                        magic_now(
                            X=X,
                            y=y,
                            inner_cv_groups=groups.loc[y.index],
                            outer_cv_groups=groups.loc[y.index],
                            scoring="FNFP",
                            algorithm=algorithm,
                            scaler="robust",
                            feature_selector="optimize",
                            hyperparameters="optimize",
                            pos_weight_factor=1.2,
                            n_trials=args.n_trials,
                            calibration="auto",
                            optimize_threshold=True,
                            output_dir=output_dir,
                            tag=f"_{protein}_{algorithm}_{target}",
                            plots=True,
                            n_jobs=12,
                        )

                        with open(log_path, "w") as file:
                            file.write("done")

            jobs.append(optimisation)

    schedule(
        *jobs,
        name="Haramo4PredOmics",
        backend=args.backend,
        prune=True,
        env=[
            "source ~/tabml/bin/activate",
        ],
    )
