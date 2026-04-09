#!/usr/bin/env python3
"""Bootstrap Mobile-O-SFT into the repo's image source-manifest format."""

from mobile_o_bootstrap_common import run_bootstrap


def main() -> None:
    run_bootstrap(
        description=__doc__ or "Bootstrap Mobile-O-SFT into the repo manifest format.",
        default_repo_id="Amshaker/Mobile-O-SFT",
        default_output_root="data/mobile_o_sft",
        default_dataset_name="mobile_o_sft",
        default_filenames="object_2.tar",
    )


if __name__ == "__main__":
    main()
