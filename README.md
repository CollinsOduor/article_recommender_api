# Next Article Recommender API

FastAPI wrapper for `NextArticleModel` in the article-level notebook. It loads the saved Keras model, scikit-learn preprocessor, article/section mappings, and titles at startup.

## Prepare artifacts

Run the article-level notebook through its model-save cell. It should produce:

- `article_recommender_article_level.keras`
- `article_recommender_article_level_preprocessor.joblib`
- `article_recommender_article_level_metadata.json`

Copy those files into this folder's `artifacts/` directory, or set `MODEL_ARTIFACTS_DIR` to their directory. The API reports `503 not_ready` until the artifacts are present. The notebook's exporter uses a picklable boolean transformer and records the section aliases required by the API; rerun the notebook's save cell after those notebook changes.

The joblib preprocessor is a pickle-based artifact and may require `dill` to load.
Load only model artifacts you trust.

## Install and run

Use the same TensorFlow, scikit-learn, and NumPy versions used to train the model; newer notebook exports record these in the metadata. The requirements pin scikit-learn to 1.6.1 for the supplied preprocessor; update the pin if you export an artifact with a different version.
Use a Python release supported by that TensorFlow version.

```bash
cd article_recommender_api
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `/docs` for the interactive OpenAPI page. `GET /health` reports whether model artifacts loaded.

For Docker, rebuild the image after changing dependencies so the container receives
the updated packages:

```bash
docker build -t article-recommender-api .
docker run --rm -p 8000:8000 article-recommender-api
```

## Find articles by title

`GET /articles/search?q=climate&limit=8` performs a case-insensitive title
search against the saved catalog and returns matching model article IDs. Only
articles supported by the loaded model are included. The notebook exports both
`article_titles` (ID to title) and `title_to_article_ids` (title to a list of IDs),
so articles sharing a title remain separate search results. Older metadata without
the reverse index is supported by deriving it from `article_titles`.

## Recommend

`POST /recommend` expects at least the model's five prior interactions, oldest first. The request may include more; only the latest five are used. `page_section` can be a raw section from the notebook's taxonomy or an already-normalized section. The recommendation timestamp must include a timezone offset because its local weekday and hour are model features.
Profile fields may be omitted or set to `null`; the saved preprocessor imputes missing values using its fitted settings.
The exported taxonomy includes both HTML-escaped and decoded aliases, such as
`Life &amp; Style` and `Life & Style`.

```json
{
  "history": [
    {"article_id": "article-100", "page_section": "News"},
    {"article_id": "article-221", "page_section": "Sports"},
    {"article_id": "article-105", "page_section": "Business"},
    {"article_id": "article-310", "page_section": "Life & Style"},
    {"article_id": "article-415", "page_section": "News"}
  ],
  "profile": {
    "age": 34,
    "gender": "female",
    "jobtitle": "Teacher",
    "subscription_status": "registered",
    "allow_marketing": false
  },
  "recommendation_time": "2026-11-20T09:00:00+03:00",
  "top_k": 5,
  "exclude_seen": true
}
```

The response contains ranked article IDs, saved titles where available, model scores, and the number of eligible candidates. Recommendations are limited to article IDs observed as training targets; this model does not provide cold-start recommendations for new articles.

## Artifact compatibility

The supplied artifacts predate the current exporter: their metadata lacks
`section_mapping`, and their preprocessor contains a serialized lambda that fails
when transformed in the Python 3.11 container. Regenerate all three artifacts by
running the current notebook before starting the API.
