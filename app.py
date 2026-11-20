from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from tensorflow.keras import Model, layers, saving


@saving.register_keras_serializable(package="article_recommender")
class NextArticleModel(Model):
    """The notebook's article-history GRU and profile model."""

    def __init__(
        self,
        num_articles: int,
        num_sections: int,
        profile_dim: int,
        article_embedding_dim: int = 32,
        section_embedding_dim: int = 8,
        gru_units: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_articles = num_articles
        self.num_sections = num_sections
        self.profile_dim = profile_dim
        self.article_embedding_dim = article_embedding_dim
        self.section_embedding_dim = section_embedding_dim
        self.gru_units = gru_units

        self.article_embedding = layers.Embedding(
            num_articles + 1, article_embedding_dim, name="article_embedding"
        )
        self.section_embedding = layers.Embedding(
            num_sections + 1, section_embedding_dim, name="section_embedding"
        )
        self.history_encoder = layers.GRU(gru_units, name="history_encoder")
        self.profile_mlp = tf.keras.Sequential(
            [
                layers.Dense(64, activation="gelu"),
                layers.LayerNormalization(),
                layers.Dropout(0.15),
            ],
            name="profile_tower",
        )
        self.fusion_mlp = tf.keras.Sequential(
            [
                layers.Dense(96, activation="gelu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="gelu"),
            ],
            name="fusion",
        )
        self.classifier = layers.Dense(num_articles, name="article_logits")

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        article_history, section_history, profile = inputs
        article_vectors = self.article_embedding(article_history)
        section_vectors = self.section_embedding(section_history)
        history_vectors = tf.concat([article_vectors, section_vectors], axis=-1)
        history_context = self.history_encoder(history_vectors, training=training)
        profile_context = self.profile_mlp(profile, training=training)
        fused = tf.concat([history_context, profile_context], axis=-1)
        return self.classifier(self.fusion_mlp(fused, training=training))

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_articles": self.num_articles,
                "num_sections": self.num_sections,
                "profile_dim": self.profile_dim,
                "article_embedding_dim": self.article_embedding_dim,
                "section_embedding_dim": self.section_embedding_dim,
                "gru_units": self.gru_units,
            }
        )
        return config


@dataclass(frozen=True)
class RecommenderResources:
    model: NextArticleModel
    preprocessor: Any
    article_to_index: dict[str, int]
    index_to_article: dict[int, str]
    article_titles: dict[str, str | None]
    title_to_article_ids: dict[str, list[str]]
    section_to_index: dict[str, int]
    section_mapping: dict[str, str]
    look_back: int
    profile_cols: list[str]
    training_versions: dict[str, str]


BASE_DIR = Path(__file__).resolve().parent
TIME_FEATURES = ("dow_sin", "dow_cos", "hour_sin", "hour_cos")


def _artifact_dir() -> Path:
    configured = Path(os.environ.get("MODEL_ARTIFACTS_DIR", "artifacts"))
    return configured if configured.is_absolute() else BASE_DIR / configured


def load_resources() -> RecommenderResources:
    artifact_dir = _artifact_dir()
    model_path = artifact_dir / "article_recommender_article_level.keras"
    preprocessor_path = artifact_dir / "article_recommender_article_level_preprocessor.joblib"
    metadata_path = artifact_dir / "article_recommender_article_level_metadata.json"
    missing = [str(path) for path in (model_path, preprocessor_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing recommender artifacts: " + ", ".join(missing))

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    required_keys = {
        "article_to_index",
        "article_titles",
        "section_to_index",
        "section_mapping",
        "look_back",
        "profile_cols",
    }
    absent_keys = required_keys.difference(metadata)
    if absent_keys:
        raise ValueError(f"Artifact metadata is missing required fields: {sorted(absent_keys)}")

    article_to_index = {
        str(article_id): int(index)
        for article_id, index in metadata["article_to_index"].items()
    }
    section_to_index = {
        str(section): int(index)
        for section, index in metadata["section_to_index"].items()
    }
    if sorted(article_to_index.values()) != list(range(len(article_to_index))):
        raise ValueError("article_to_index must contain contiguous indices starting at zero")
    if sorted(section_to_index.values()) != list(range(len(section_to_index))):
        raise ValueError("section_to_index must contain contiguous indices starting at zero")

    article_titles = {
        str(article_id): (str(title) if title is not None else None)
        for article_id, title in metadata["article_titles"].items()
    }
    title_to_article_ids: dict[str, list[str]] = {}
    for article_id, title in article_titles.items():
        if title:
            title_to_article_ids.setdefault(title, []).append(article_id)
    if "title_to_article_ids" in metadata:
        if metadata["title_to_article_ids"] != title_to_article_ids:
            raise ValueError("Saved title-to-article mapping does not match article_titles")
        title_to_article_ids = metadata["title_to_article_ids"]

    preprocessor = joblib.load(preprocessor_path)
    profile_cols = list(metadata["profile_cols"])
    expected_profile_cols = list(preprocessor.get_feature_names_out()) + list(TIME_FEATURES)
    if profile_cols != expected_profile_cols:
        raise ValueError("Saved profile columns do not match the saved preprocessor feature order")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "NextArticleModel": NextArticleModel,
            "article_recommender>NextArticleModel": NextArticleModel,
        },
    )
    if model.num_articles != len(article_to_index):
        raise ValueError("Model article output size does not match article_to_index")
    if model.num_sections != len(section_to_index):
        raise ValueError("Model section vocabulary size does not match section_to_index")
    if model.profile_dim != len(profile_cols):
        raise ValueError("Model profile input size does not match saved profile_cols")

    return RecommenderResources(
        model=model,
        preprocessor=preprocessor,
        article_to_index=article_to_index,
        index_to_article={index: article_id for article_id, index in article_to_index.items()},
        article_titles=article_titles,
        title_to_article_ids=title_to_article_ids,
        section_to_index=section_to_index,
        section_mapping={
            str(section): str(mapped_section)
            for section, mapped_section in metadata["section_mapping"].items()
        },
        look_back=int(metadata["look_back"]),
        profile_cols=profile_cols,
        training_versions=dict(metadata.get("training_versions", {})),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.resources = load_resources()
        app.state.load_error = None
    except FileNotFoundError as error:
        app.state.resources = None
        app.state.load_error = str(error)
    yield


app = FastAPI(
    title="Next Article Recommender",
    version="1.0.0",
    description="Ranks train-seen articles using the notebook's recent-history model.",
    lifespan=lifespan,
)


class UserProfile(BaseModel):
    age: float | None = Field(default=None, ge=0)
    gender: str | None = None
    jobtitle: str | None = None
    subscription_status: str | None = None
    allow_marketing: bool | None = None


class HistoryEvent(BaseModel):
    article_id: str = Field(min_length=1, max_length=256)
    page_section: str | None = Field(default=None, max_length=128)

    @field_validator("article_id")
    @classmethod
    def strip_article_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("article_id must not be blank")
        return value

    @field_validator("page_section")
    @classmethod
    def strip_page_section(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class RecommendationRequest(BaseModel):
    history: list[HistoryEvent] = Field(min_length=1, max_length=500)
    profile: UserProfile
    recommendation_time: datetime
    top_k: int = Field(default=10, ge=1, le=50)
    exclude_seen: bool = False

    @field_validator("recommendation_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recommendation_time must include a timezone offset")
        return value


class Recommendation(BaseModel):
    article_id: str
    title: str | None
    score: float


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation]
    history_used: int
    candidate_count: int


class ArticleSearchResult(BaseModel):
    article_id: str
    title: str


def _profile_vector(
    resources: RecommenderResources,
    profile: UserProfile,
    recommendation_time: datetime,
) -> np.ndarray:
    profile_frame = pd.DataFrame([
        {key: np.nan if value is None else value for key, value in profile.model_dump().items()}
    ])
    transformed = resources.preprocessor.transform(profile_frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float32).reshape(-1)
    feature_names = list(resources.preprocessor.get_feature_names_out())
    if len(feature_names) != transformed.size:
        raise RuntimeError("Preprocessor output width does not match its feature names")

    weekday = recommendation_time.weekday()
    hour = recommendation_time.hour
    feature_values = dict(zip(feature_names, transformed, strict=True))
    feature_values.update(
        {
            "dow_sin": float(np.sin(2 * np.pi * weekday / 7)),
            "dow_cos": float(np.cos(2 * np.pi * weekday / 7)),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        }
    )
    missing = [column for column in resources.profile_cols if column not in feature_values]
    if missing:
        raise RuntimeError(f"Unable to build model profile features: {missing}")
    return np.asarray(
        [feature_values[column] for column in resources.profile_cols],
        dtype=np.float32,
    )[None, :]


@app.get("/articles/search", response_model=list[ArticleSearchResult])
def search_articles(
    request: Request,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=8, ge=1, le=20),
) -> list[ArticleSearchResult]:
    resources: RecommenderResources | None = request.app.state.resources
    if resources is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": request.app.state.load_error,
            },
        )

    query = q.strip().casefold()
    if not query:
        raise HTTPException(status_code=422, detail="Search query must not be blank.")

    matches = [
        ArticleSearchResult(article_id=article_id, title=title)
        for title, article_ids in resources.title_to_article_ids.items()
        if query in title.casefold()
        for article_id in article_ids
        if article_id in resources.article_to_index
    ]
    matches.sort(
        key=lambda article: (
            not article.title.casefold().startswith(query),
            len(article.title),
            article.title.casefold(),
            article.article_id,
        )
    )
    return matches[:limit]


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    resources: RecommenderResources | None = request.app.state.resources
    if resources is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": request.app.state.load_error,
            },
        )
    return {
        "status": "ready",
        "candidate_articles": len(resources.article_to_index),
        "look_back": resources.look_back,
        "training_versions": resources.training_versions,
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
    resources: RecommenderResources | None = request.app.state.resources
    if resources is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "reason": request.app.state.load_error,
            },
        )
    if len(payload.history) < resources.look_back:
        raise HTTPException(
            status_code=422,
            detail=f"Provide at least {resources.look_back} history events, oldest to newest.",
        )

    recent = payload.history[-resources.look_back :]
    article_oov = len(resources.article_to_index)
    section_oov = len(resources.section_to_index)
    article_history = np.asarray(
        [
            resources.article_to_index.get(event.article_id, article_oov)
            for event in recent
        ],
        dtype=np.int32,
    )[None, :]
    section_history = np.asarray(
        [
            resources.section_to_index.get(
                resources.section_mapping.get(event.page_section, event.page_section),
                section_oov,
            )
            if event.page_section
            else section_oov
            for event in recent
        ],
        dtype=np.int32,
    )[None, :]
    profile = _profile_vector(resources, payload.profile, payload.recommendation_time)
    logits = resources.model(
        [article_history, section_history, profile],
        training=False,
    ).numpy()[0]
    if logits.shape != (len(resources.article_to_index),):
        raise RuntimeError("Model returned an unexpected number of article scores")
    if not np.isfinite(logits).all():
        raise RuntimeError("Model returned non-finite article scores")

    if payload.exclude_seen:
        seen_indices = {
            resources.article_to_index[event.article_id]
            for event in recent
            if event.article_id in resources.article_to_index
        }
        if seen_indices:
            logits[list(seen_indices)] = -np.inf

    available = int(np.isfinite(logits).sum())
    if available == 0:
        return RecommendationResponse(
            recommendations=[],
            history_used=len(recent),
            candidate_count=0,
        )
    stable_logits = np.where(np.isfinite(logits), logits, -np.inf)
    probabilities = np.exp(stable_logits - np.max(stable_logits))
    probabilities /= probabilities.sum()
    indices = np.argsort(logits)[::-1][: min(payload.top_k, available)]

    recommendations = [
        Recommendation(
            article_id=resources.index_to_article[int(index)],
            title=resources.article_titles.get(resources.index_to_article[int(index)]),
            score=float(probabilities[index]),
        )
        for index in indices
    ]
    return RecommendationResponse(
        recommendations=recommendations,
        history_used=len(recent),
        candidate_count=available,
    )
