import argparse
import logging
from pathlib import Path

from .part_spec import parse_parts_spec
from .paths import DataLayout


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _common_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Mini-repo root directory (default: download_data folder)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openvid-dataops",
        description="Download OpenVid parts and preprocess using WAN VAE",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download selected OpenVid parts (+ optional extract)")
    _common_root_arg(p_dl)
    p_dl.add_argument(
        "--parts",
        required=True,
        help="Part spec, e.g. [1,2,4,5], 1,2,4,5, or 'all'",
    )
    p_dl.add_argument(
        "--part-index-base",
        type=int,
        default=0,
        choices=[0, 1],
        help="Input part numbering base. Default 0 means user part N maps to remote part N.",
    )
    p_dl.add_argument("--extract", action="store_true", help="Extract zip after download")
    p_dl.add_argument("--keep-zip", action="store_true", help="Keep zip files after extraction")
    p_dl.add_argument("--no-csv", action="store_true", help="Do not download OpenVid CSV")
    p_dl.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parts to download/extract in parallel within one process (default: 1)",
    )

    p_mf = sub.add_parser("build-manifest", help="Build manifest from extracted parts + OpenVid CSV")
    _common_root_arg(p_mf)
    p_mf.add_argument("--parts", default=None, help="Optional part spec filter, supports 'all'")
    p_mf.add_argument(
        "--part-index-base",
        type=int,
        default=0,
        choices=[0, 1],
        help="Part numbering base used in --parts",
    )
    p_mf.add_argument("--output-name", default="openvid_selected_parts.csv")

    p_en = sub.add_parser("encode", help="Encode manifest videos using WAN VAE")
    _common_root_arg(p_en)
    p_en.add_argument("--manifest-csv", type=Path, default=None, help="Manifest CSV path")
    p_en.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints/wan/wanxiang1_3b"),
        help="WAN ckpt dir under download_data, default: checkpoints/wan/wanxiang1_3b",
    )
    p_en.add_argument("--task", default="t2v-1.3B", help="WAN task key")
    p_en.add_argument("--frame-num", type=int, default=81)
    p_en.add_argument("--sampling-rate", type=int, default=1)
    p_en.add_argument("--skip-num", type=int, default=0)
    p_en.add_argument("--target-size", default="480,832", help="H,W")
    p_en.add_argument("--max-samples", type=int, default=None)
    p_en.add_argument("--log-every", type=int, default=25)
    p_en.add_argument("--output-subdir", default="wan_vae", help="Subdir under data/openvid/encoded")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    layout = DataLayout(root=args.root.resolve())

    if args.command == "download":
        from .download import download_openvid

        parts = parse_parts_spec(args.parts, allow_all=True)
        download_openvid(
            layout=layout,
            parts_user=parts,
            part_index_base=args.part_index_base,
            extract=args.extract,
            keep_zip=args.keep_zip,
            include_csv=(not args.no_csv),
            jobs=args.jobs,
        )
        return

    if args.command == "build-manifest":
        from .manifest import build_manifest

        parts = parse_parts_spec(args.parts, allow_all=True) if args.parts else None
        build_manifest(
            layout=layout,
            selected_parts_user=parts,
            part_index_base=args.part_index_base,
            output_name=args.output_name,
        )
        return

    if args.command == "encode":
        from .encode import encode_manifest

        if args.manifest_csv is None:
            manifest_csv = layout.manifests_root / "openvid_selected_parts.csv"
        else:
            manifest_csv = args.manifest_csv
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

        h, w = [int(x.strip()) for x in args.target_size.split(",")]
        output_dir = layout.encoded_root / args.output_subdir
        encode_manifest(
            ckpt_dir=args.ckpt_dir.resolve(),
            manifest_csv=manifest_csv.resolve(),
            output_dir=output_dir.resolve(),
            task=args.task,
            frame_num=args.frame_num,
            sampling_rate=args.sampling_rate,
            skip_num=args.skip_num,
            target_size=(h, w),
            max_samples=args.max_samples,
            log_every=args.log_every,
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
