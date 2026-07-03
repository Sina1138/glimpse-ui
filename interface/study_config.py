"""Study configuration for ReView evaluation conditions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent

# Study dataset: produced by pipeline/preprocess_study_papers.py from the
# frozen review texts in study/papers_raw/. Kept separate from the demo
# dataset in data/ so study builds never show demo papers.
STUDY_DATA_CSV = BASE_DIR / "study" / "study_data" / "study_scored_reviews.csv"


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for a study condition."""
    condition: str                    # "full" or "no_highlight"
    logging_enabled: bool             # True for study variants
    log_dir: Path                     # directory for JSONL interaction logs
    study_mode: bool = False          # True: session gate (participant ID + task
                                      # start/end), study dataset only, no
                                      # Interactive tab, no year/paper navigation
    data_csv: Optional[Path] = None   # dataset override; None = demo dataset

    @property
    def highlights_enabled(self) -> bool:
        return self.condition == "full"


def default_config() -> StudyConfig:
    """Default config: full features, no logging (backward-compat Demo.py)."""
    return StudyConfig(
        condition="full",
        logging_enabled=False,
        log_dir=BASE_DIR / "study" / "interaction_logs",
    )


def full_study_config() -> StudyConfig:
    """Full (ReView) study condition: logging + session gate + study dataset."""
    return StudyConfig(
        condition="full",
        logging_enabled=True,
        log_dir=BASE_DIR / "study" / "interaction_logs",
        study_mode=True,
        data_csv=STUDY_DATA_CSV,
    )


def no_highlight_study_config() -> StudyConfig:
    """No Highlighting study condition: logging + session gate + study dataset."""
    return StudyConfig(
        condition="no_highlight",
        logging_enabled=True,
        log_dir=BASE_DIR / "study" / "interaction_logs",
        study_mode=True,
        data_csv=STUDY_DATA_CSV,
    )
