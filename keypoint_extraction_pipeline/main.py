import logging
import os
import sys
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig
import ray
from datasets import load_dataset
from ray.data import Dataset

from keypoint_extraction_pipeline.savers.saver import (
    BaseSaver,
)
from keypoint_extraction_pipeline.batch_processor import BatchProcessor
from keypoint_extraction_pipeline.transformations.pipeline_processor import (
    PipelineProcessor,
)


# Logger config
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def _init_ray(cfg: DictConfig):
    """Helper to init Ray."""
    from ray.data.context import DatasetContext

    ctx = DatasetContext.get_current()
    ctx.target_max_block_size = 2 * 1024 * 1024  # 2 MiB  → bloques pequeños

    if not ray.is_initialized():
        logger.info("Initializing Ray...")
        ray.init(
            runtime_env={"working_dir": "."},
            log_to_driver=cfg.ray.get("log_to_driver", True),
            logging_level=logging.INFO,
            num_cpus=cfg.ray.get("num_cpus", None),
            include_dashboard=cfg.ray.get("include_dashboard", False),
            dashboard_host=cfg.ray.get("dashboard_host", "0.0.0.0"),
            dashboard_port=cfg.ray.get("dashboard_port", 8265),
        )
        logger.info(f"Available resources: {ray.cluster_resources()}")
    else:
        logger.info("Ray is already initialized.")


def _shutdown_ray():
    """Helper para apagar Ray si está inicializado."""
    if ray.is_initialized():
        logger.info("Shutting down Ray...")
        ray.shutdown()


def _process_raw_keypoints_logic(cfg: DictConfig):
    """Processes and saves raw keypoints using given config."""
    logger.info("Starting raw keypoint processing...")
    BASE_PATH = Path.cwd()

    # Load HF dataset dataset using a temporary directory as cache
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        dataset = load_dataset(
            path=f"{BASE_PATH}/keypoint_extraction_pipeline/loaders/{cfg.dataset.loader_name}.py",
            data_dir=cfg.dataset.dataset_path,
            cache_dir=temp_cache_dir,
            trust_remote_code=True,
        )

    # get ray config
    splits = cfg.dataset.get("splits_to_process", ["train"])
    batch_size = cfg.ray.batch_size
    available_cpus = int(ray.cluster_resources().get("CPU", 1))
    num_workers = cfg.ray.num_actors or max(1, int(available_cpus) - 1)
    logger.info(f"Using {num_workers} worker actors for keypoint extraction.")

    # iterate through splits and process each one with Ray
    for split in splits:
        logger.info(f"Processing split: {split}")
        hf_split = dataset[split]  # type: ignore
        if cfg.dataset.max_samples:
            hf_split = hf_split.select(
                range(min(len(hf_split), cfg.dataset.max_samples))
            )

        ds: Dataset = ray.data.from_huggingface(hf_split)
        if ds.num_blocks() < 2 * num_workers:
            ds = ds.repartition(2 * num_workers)
        # .repartition(num_workers * 2)

        # process
        processed_ds = ds.map_batches(
            BatchProcessor,  # type: ignore
            fn_constructor_kwargs={
                "extractor_config": cfg.extractor,
                "frame_extractor_config": cfg.frame_extractor,
            },
            batch_size=batch_size,
            concurrency=num_workers,
            num_cpus=1,  # CPU per actor/task
            batch_format="numpy",
        )

        save_dir = Path(cfg.dataset.raw_keypoints_save_dir) / split
        saver_instance = hydra.utils.instantiate(cfg.saver, save_dir=save_dir)
        logger.info(
            f"Using saver: {saver_instance.__class__.__name__} for raw keypoints."
        )

        # save
        processed_ds.map_batches(
            saver_instance,
            batch_size=1,
            concurrency=num_workers,
            num_cpus=1,
            zero_copy_batch=True,  # avoid copying the batch
        ).materialize()

        logger.info(f"Finished split '{split}', results in {save_dir}")
    logger.info("Raw keypoint processing completed.")


def _transform_keypoints_logic(cfg: DictConfig):
    """Processed and save saved keypoints applying transformations to them."""
    logger.info("Starting keypoint transformation...")

    input_dir = Path(cfg.dataset.raw_keypoints_save_dir)
    splits = cfg.dataset.get("splits_to_process", ["train"])

    # get ray config
    available_cpus = ray.cluster_resources().get("CPU", 1)
    num_actors_transform = cfg.ray.get("num_actors_transform", cfg.ray.num_actors)
    num_workers = num_actors_transform or max(1, int(available_cpus) - 1)
    logger.info(f"Using {num_workers} worker actors for keypoint transformation.")

    # iterate through splits and process each one with Ray
    for split_name in splits:
        logger.info(f"Transforming split: {split_name}")
        split_save_dir = os.path.join(cfg.pipeline.transformed_kps_save_dir, split_name)

        temp_saver_for_ext = hydra.utils.instantiate(cfg.saver, save_dir="dummy")
        save_extension = temp_saver_for_ext.get_file_extension()

        split_path = Path.joinpath(input_dir, split_name)
        logger.info(
            f"Loading files with extension {save_extension} for split {split_name}, path: {split_path}..."
        )
        files = [str(p) for p in split_path.glob(f"*{save_extension}")]

        if not files:
            logger.warning(
                f"No files found for split '{split_name}'. Skipping this split."
            )
            continue

        split_ds: Dataset = ray.data.from_items([{"file_path": p} for p in files])

        output_saver_class = hydra.utils.get_class(cfg.saver._target_)
        output_saver_instance: BaseSaver = output_saver_class(save_dir=split_save_dir)
        logger.info(
            f"Using saver: {output_saver_instance.__class__.__name__} for transformed keypoints."
        )

        pipeline_instance = PipelineProcessor(
            cfg.pipeline, saver_class=output_saver_class  # type: ignore
        )

        processed = split_ds.map_batches(
            pipeline_instance,
            batch_size=cfg.ray.batch_size,
            concurrency=num_workers,
            num_cpus=1,
        )

        processed.map_batches(
            output_saver_instance,
            batch_size=1,  # a file per processed item
            concurrency=num_workers,
            num_cpus=1,
        ).materialize()
        logger.info(
            f"Finished transforming split '{split_name}', results in {split_save_dir}"
        )

    logger.info("Keypoint transformation completed.")


def _transform_single_keypoints_logic(cfg: DictConfig):
    logger.info("Starting single-video keypoint transformation…")

    input_dir = Path(cfg.single_video.raw_output_path)
    output_dir = Path(cfg.single_video.transformed_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_saver = hydra.utils.instantiate(cfg.saver, save_dir="dummy")
    ext = temp_saver.get_file_extension()

    files = [str(p) for p in input_dir.glob(f"*{ext}")]
    if not files:
        logger.warning(f"No keypoint files found in {input_dir}. Nothing to transform.")
        return

    available_cpus = ray.cluster_resources().get("CPU", 1)
    num_workers = min(len(files), int(available_cpus)) or 1

    ds: Dataset = ray.data.from_items([{"file_path": p} for p in files])

    saver_cls = hydra.utils.get_class(cfg.saver._target_)
    saver_out: BaseSaver = saver_cls(save_dir=output_dir)
    pipeline_proc = PipelineProcessor(cfg.pipeline, saver_class=saver_cls)  # type: ignore

    processed = (
        ds.map_batches(
            pipeline_proc,
            batch_size=cfg.ray.batch_size,
            concurrency=num_workers,
            num_cpus=1,
        )
        .map_batches(
            saver_out,
            batch_size=1,
            concurrency=num_workers,
            num_cpus=1,
            zero_copy_batch=True,
        )
        .materialize()
    )

    logger.info(f"Single-video transformation completed. Results in {output_dir}")


def _process_single_video(cfg: DictConfig):
    item = {
        "segment_id": Path(cfg.single_video.path).stem,
        "video_path": cfg.single_video.path,
        "start_time_ms": cfg.single_video.start_time_ms,
        "end_time_ms": cfg.single_video.end_time_ms,
        "label": cfg.single_video.label,
    }
    ds = ray.data.from_items([item])

    processed_ds = ds.map_batches(
        BatchProcessor,
        fn_constructor_kwargs={
            "extractor_config": cfg.extractor,
            "frame_extractor_config": cfg.frame_extractor,
        },
        batch_size=1,
        concurrency=1,
        num_cpus=1,
        batch_format="numpy",
    )

    save_dir = Path(cfg.single_video.raw_output_path)
    saver_instance = hydra.utils.instantiate(cfg.saver, save_dir=save_dir)

    processed_ds.map_batches(
        saver_instance,
        batch_size=1,
        concurrency=1,
        num_cpus=1,
        zero_copy_batch=True,
    ).materialize()

    logger.info(f"Vídeo procesado. Resultado en {save_dir}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_pipeline(cfg: DictConfig):
    """
    Main orchestrator for the pipeline.
    Run steps following configured mode.
    """
    mode = cfg.mode
    if not mode:
        raise ValueError(
            "Mode not specified in config. Use 'extract', 'transform' or 'all'."
        )

    ray_initialized_by_orchestrator = False

    try:
        if mode == "extract" or mode == "all":
            if not ray.is_initialized():
                _init_ray(cfg)
                ray_initialized_by_orchestrator = True
            _process_raw_keypoints_logic(cfg)

        elif mode == "transform" or mode == "all":
            if not ray.is_initialized():
                _init_ray(cfg)
                ray_initialized_by_orchestrator = True
            elif mode == "all" and not ray_initialized_by_orchestrator:
                pass

            _transform_keypoints_logic(cfg)

        elif mode == "single":
            if not ray.is_initialized():
                _init_ray(cfg)
                ray_initialized_by_orchestrator = True

            _process_single_video(cfg)

            if cfg.single_video.apply_transformations:
                if getattr(cfg, "pipeline", None):
                    _transform_single_keypoints_logic(cfg)

        else:
            logger.error(
                f"Invalid mode: '{mode}'. Choose from 'extract', 'transform', 'all', 'single'."
            )

    finally:
        if ray_initialized_by_orchestrator:
            _shutdown_ray()
        elif ray.is_initialized():
            logger.info(
                "Ray was initialized externally, not shutting down from orchestrator."
            )


if __name__ == "__main__":
    run_pipeline()
